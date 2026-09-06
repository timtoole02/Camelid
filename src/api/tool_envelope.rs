//! Recognising (and undoing) a model that echoes the JSON-Schema envelope it was
//! shown instead of the arguments that schema describes.
//!
//! A tool is advertised to the model as a JSON Schema:
//!
//! ```json
//! {"name": "get_weather",
//!  "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}
//! ```
//!
//! Small models frequently reply with the *shape they were shown* rather than an
//! instance of it, emitting `{"properties": {"city": "Paris"}}` where
//! `{"city": "Paris"}` was meant. Relaying that verbatim is wire-correct but
//! useless: an OpenAI client reads `arguments.city` and finds nothing.
//!
//! The repair here is deliberately narrow. It never guesses, never invents a
//! value, and only unwraps when the *declared* schema proves the wrapper cannot
//! itself be a real argument. Anything ambiguous is passed through untouched, so
//! the worst case is the behaviour we already had.

use std::collections::{BTreeSet, HashMap};

use serde_json::Value;

/// Keys that may appear beside `properties` in an echoed JSON-Schema envelope.
///
/// An arguments object made only of these (plus `properties`) cannot be a
/// meaningful set of arguments, which is what makes the unwrap unambiguous.
const SCHEMA_ENVELOPE_KEYS: &[&str] = &[
    "$defs",
    "$schema",
    "additionalProperties",
    "definitions",
    "description",
    "properties",
    "required",
    "title",
    "type",
];

/// The literal key a schema-echoing model wraps its arguments in.
const PROPERTIES_KEY: &str = "properties";

/// The top-level parameter names each declared tool accepts, keyed by function name.
///
/// Absent function name means "this tool was never declared"; a present but empty
/// set means "declared, and it takes no parameters". The two are treated
/// differently on purpose: only a declared schema can authorise an unwrap.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct ToolParameterNames(HashMap<String, BTreeSet<String>>);

impl ToolParameterNames {
    /// Read the declared parameter names out of an OpenAI `tools` array.
    ///
    /// Tolerates both the nested (`{"function": {"name", "parameters"}}`) and the
    /// flat (`{"name", "parameters"}`) spellings; anything unrecognised is skipped
    /// rather than rejected, because this type only ever *withholds* permission to
    /// rewrite and so cannot make the response worse by knowing less.
    pub(crate) fn from_request_tools(tools: &[Value]) -> Self {
        let mut declared: HashMap<String, BTreeSet<String>> = HashMap::new();
        for tool in tools {
            let Some(obj) = tool.as_object() else {
                continue;
            };
            let scope = obj
                .get("function")
                .and_then(Value::as_object)
                .unwrap_or(obj);
            let Some(name) = scope.get("name").and_then(Value::as_str) else {
                continue;
            };
            let name = name.trim();
            if name.is_empty() {
                continue;
            }
            let names = scope
                .get("parameters")
                .and_then(Value::as_object)
                .and_then(|parameters| parameters.get(PROPERTIES_KEY))
                .and_then(Value::as_object)
                .map(|properties| properties.keys().cloned().collect())
                .unwrap_or_default();
            declared.insert(name.to_string(), names);
        }
        Self(declared)
    }

    fn declared_for(&self, function: &str) -> Option<&BTreeSet<String>> {
        self.0.get(function.trim())
    }
}

/// Undo a schema envelope the model echoed around its arguments.
///
/// Returns `args` unchanged unless every one of these holds:
///
/// 1. `args` is an object containing `properties`, and every other key it has is
///    an envelope key that the tool does **not** declare as a parameter (a key
///    the tool declares is a real argument, however envelope-like it looks);
/// 2. the tool was actually declared, and does **not** declare a parameter named
///    `properties` (otherwise the wrapper could legitimately be an argument);
/// 3. `properties` holds a non-empty object whose every key is a parameter the
///    tool declares.
///
/// Condition 3 is what keeps this from laundering a hallucination: if the model
/// invented a field, nothing is unwrapped and the caller sees exactly what the
/// model produced.
pub(crate) fn unwrap_schema_envelope(
    function: &str,
    args: Value,
    declared: &ToolParameterNames,
) -> Value {
    let Some(obj) = args.as_object() else {
        return args;
    };
    if !obj.contains_key(PROPERTIES_KEY) {
        return args;
    }
    let Some(names) = declared.declared_for(function) else {
        return args;
    };
    if names.contains(PROPERTIES_KEY) {
        return args;
    }
    // `type`, `title`, `description` and `required` are ordinary parameter names
    // as well as schema keywords; when the tool declares one it is data, not noise.
    if !obj.keys().all(|key| {
        key == PROPERTIES_KEY
            || (SCHEMA_ENVELOPE_KEYS.contains(&key.as_str()) && !names.contains(key))
    }) {
        return args;
    }
    let Some(inner) = obj.get(PROPERTIES_KEY).and_then(Value::as_object) else {
        return args;
    };
    if inner.is_empty() || !inner.keys().all(|key| names.contains(key)) {
        return args;
    }
    Value::Object(inner.clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn weather_tool() -> Value {
        json!({
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"]
                }
            }
        })
    }

    fn currency_tool() -> Value {
        json!({
            "type": "function",
            "function": {
                "name": "convert_currency",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "amount": {"type": "number"},
                        "from_currency": {"type": "string"},
                        "to_currency": {"type": "string"}
                    }
                }
            }
        })
    }

    fn declared() -> ToolParameterNames {
        ToolParameterNames::from_request_tools(&[weather_tool(), currency_tool()])
    }

    #[test]
    fn the_observed_wrapper_is_unwrapped() {
        // Exactly the shape Camelid returned for "What is the weather in Paris?".
        let args = json!({"properties": {"city": "Paris"}});
        let repaired = unwrap_schema_envelope("get_weather", args, &declared());
        assert_eq!(repaired, json!({"city": "Paris"}));
    }

    #[test]
    fn a_full_schema_echo_is_unwrapped() {
        let args = json!({
            "type": "object",
            "required": ["city"],
            "properties": {"city": "Tokyo"}
        });
        let repaired = unwrap_schema_envelope("get_weather", args, &declared());
        assert_eq!(repaired, json!({"city": "Tokyo"}));
    }

    #[test]
    fn every_declared_parameter_survives_the_unwrap() {
        let args = json!({
            "properties": {"amount": 100, "from_currency": "USD", "to_currency": "EUR"}
        });
        let repaired = unwrap_schema_envelope("convert_currency", args, &declared());
        assert_eq!(
            repaired,
            json!({"amount": 100, "from_currency": "USD", "to_currency": "EUR"})
        );
    }

    #[test]
    fn arguments_that_are_already_correct_are_untouched() {
        let args = json!({"city": "Paris"});
        let repaired = unwrap_schema_envelope("get_weather", args.clone(), &declared());
        assert_eq!(repaired, args);
    }

    #[test]
    fn unwrapping_is_idempotent() {
        let once = unwrap_schema_envelope(
            "get_weather",
            json!({"properties": {"city": "Paris"}}),
            &declared(),
        );
        let twice = unwrap_schema_envelope("get_weather", once.clone(), &declared());
        assert_eq!(once, twice);
    }

    #[test]
    fn a_tool_that_really_takes_properties_is_never_unwrapped() {
        // The wrapper is indistinguishable from a real argument here, so the
        // model's output must be relayed exactly as produced.
        let tool = json!({
            "type": "function",
            "function": {
                "name": "configure",
                "parameters": {
                    "type": "object",
                    "properties": {"properties": {"type": "object"}}
                }
            }
        });
        let declared = ToolParameterNames::from_request_tools(&[tool]);
        let args = json!({"properties": {"colour": "red"}});
        let repaired = unwrap_schema_envelope("configure", args.clone(), &declared);
        assert_eq!(repaired, args);
    }

    #[test]
    fn an_undeclared_field_blocks_the_unwrap() {
        // "continent" is not part of the schema: unwrapping would launder an
        // invention into something that looks authoritative.
        let args = json!({"properties": {"city": "Paris", "continent": "Europe"}});
        let repaired = unwrap_schema_envelope("get_weather", args.clone(), &declared());
        assert_eq!(repaired, args);
    }

    #[test]
    fn an_undeclared_tool_blocks_the_unwrap() {
        let args = json!({"properties": {"city": "Paris"}});
        let repaired = unwrap_schema_envelope("send_email", args.clone(), &declared());
        assert_eq!(repaired, args);
    }

    #[test]
    fn a_real_argument_beside_the_wrapper_blocks_the_unwrap() {
        let args = json!({"properties": {"city": "Paris"}, "city": "Oslo"});
        let repaired = unwrap_schema_envelope("get_weather", args.clone(), &declared());
        assert_eq!(repaired, args);
    }

    #[test]
    fn a_declared_parameter_that_is_also_a_schema_keyword_blocks_the_unwrap() {
        // "title" and "description" are schema keywords AND perfectly ordinary
        // parameter names. Treating this as envelope noise would silently drop
        // the title the model actually supplied.
        let tool = json!({
            "type": "function",
            "function": {
                "name": "create_issue",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "description": {"type": "string"}
                    }
                }
            }
        });
        let declared = ToolParameterNames::from_request_tools(&[tool]);
        let args = json!({"title": "Bug report", "properties": {"description": "it broke"}});
        let repaired = unwrap_schema_envelope("create_issue", args.clone(), &declared);
        assert_eq!(repaired, args);
    }

    #[test]
    fn a_schema_keyword_the_tool_does_not_declare_is_still_treated_as_envelope() {
        // get_weather declares only "city", so "required" here really is noise.
        let args = json!({"required": ["city"], "properties": {"city": "Lima"}});
        let repaired = unwrap_schema_envelope("get_weather", args, &declared());
        assert_eq!(repaired, json!({"city": "Lima"}));
    }

    #[test]
    fn a_non_object_properties_value_blocks_the_unwrap() {
        let args = json!({"properties": "city"});
        let repaired = unwrap_schema_envelope("get_weather", args.clone(), &declared());
        assert_eq!(repaired, args);
    }

    #[test]
    fn an_empty_wrapper_blocks_the_unwrap() {
        let args = json!({"properties": {}});
        let repaired = unwrap_schema_envelope("get_weather", args.clone(), &declared());
        assert_eq!(repaired, args);
    }

    #[test]
    fn non_object_arguments_are_untouched() {
        for args in [json!("Paris"), json!(7), json!(null), json!(["Paris"])] {
            let repaired = unwrap_schema_envelope("get_weather", args.clone(), &declared());
            assert_eq!(repaired, args);
        }
    }

    #[test]
    fn a_tool_with_no_declared_parameters_blocks_the_unwrap() {
        let tool = json!({"type": "function", "function": {"name": "ping"}});
        let declared = ToolParameterNames::from_request_tools(&[tool]);
        let args = json!({"properties": {"city": "Paris"}});
        let repaired = unwrap_schema_envelope("ping", args.clone(), &declared);
        assert_eq!(repaired, args);
    }

    #[test]
    fn declared_names_are_read_from_the_flat_tool_spelling() {
        let tool = json!({
            "name": "get_weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
        });
        let declared = ToolParameterNames::from_request_tools(&[tool]);
        let repaired = unwrap_schema_envelope(
            "get_weather",
            json!({"properties": {"city": "Cairo"}}),
            &declared,
        );
        assert_eq!(repaired, json!({"city": "Cairo"}));
    }

    #[test]
    fn a_surrounding_name_is_matched_after_trimming() {
        let repaired = unwrap_schema_envelope(
            "  get_weather  ",
            json!({"properties": {"city": "Oslo"}}),
            &declared(),
        );
        assert_eq!(repaired, json!({"city": "Oslo"}));
    }

    #[test]
    fn malformed_tool_entries_are_skipped_without_panicking() {
        let declared = ToolParameterNames::from_request_tools(&[
            json!("not an object"),
            json!({"function": {"name": ""}}),
            json!({"function": {}}),
            json!(null),
            weather_tool(),
        ]);
        assert!(declared.declared_for("get_weather").is_some());
        assert!(declared.declared_for("").is_none());
    }

    #[test]
    fn an_empty_tool_list_authorises_nothing() {
        let declared = ToolParameterNames::from_request_tools(&[]);
        assert!(declared.declared_for("get_weather").is_none());
        let args = json!({"properties": {"city": "Paris"}});
        assert_eq!(
            unwrap_schema_envelope("get_weather", args.clone(), &declared),
            args
        );
    }
}
