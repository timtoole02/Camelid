//! Parse a model's generated text into tool calls (Hybrid Phase 1). The server
//! renders tool definitions through the model's own chat template; this turns the
//! model's *output* back into structured calls. Family-specific: Llama 3.x emits
//! JSON (`{"name":…,"parameters":…}`, optionally `<|python_tag|>`-wrapped);
//! Qwen3/Hermes emit `<tool_call>{…}</tool_call>`. Common, unambiguous JSON
//! escaping mistakes are repaired before a malformed envelope is treated as
//! plain text; unrecoverable output still yields no calls — never a panic.

use serde_json::Value;

use super::tools::ToolCall;

/// Parse `text` into zero or more tool calls. Empty = no tool call (plain answer).
pub fn parse(text: &str, family: &str) -> Vec<ToolCall> {
    // Ornith / qwen35 emit a custom XML form `<tool_call><function=NAME>
    // <parameter=ARG>VALUE</parameter>…</function></tool_call>` (NOT JSON), so it
    // must be checked BEFORE the qwen/hermes arm (note "qwen35" contains "qwen").
    if family.contains("ornith") || family.contains("qwen35") {
        let calls = parse_ornith(text);
        if !calls.is_empty() {
            return calls;
        }
        // Fall back to hermes/JSON in case a future build emits standard tags.
        let calls = parse_hermes(text);
        if !calls.is_empty() {
            return calls;
        }
        return parse_json(text);
    }
    if family.contains("mistral") {
        let calls = parse_mistral(text);
        if !calls.is_empty() {
            return calls;
        }
        return parse_json(text);
    }
    // Gemma 4 emits `<|tool_call>call:NAME{argsâ€¦}<tool_call|>` (certified
    // serve-lane branch; the streamed content this parser sees has the
    // `<|"|>` string-quote token already stripped by detokenization, which
    // `parse_gemma4` tolerates). Matching "gemma" is safe for gemma2/gemma3
    // sessions too: those templates never emit the marker, and the JSON
    // fallbacks below are unchanged.
    if family.contains("gemma") {
        let calls = parse_gemma4(text);
        if !calls.is_empty() {
            return calls;
        }
        let calls = parse_json(text);
        if !calls.is_empty() {
            return calls;
        }
        return parse_bare_call(text);
    }
    let hermes_first = family.contains("qwen") || family.contains("hermes");
    if hermes_first {
        let calls = parse_hermes(text);
        if !calls.is_empty() {
            return calls;
        }
        let calls = parse_json(text);
        if !calls.is_empty() {
            return calls;
        }
        return parse_bare_call(text);
    }
    let calls = parse_json(text);
    if !calls.is_empty() {
        return calls;
    }
    let calls = parse_hermes(text);
    if !calls.is_empty() {
        return calls;
    }
    parse_bare_call(text)
}

/// Last resort, every family: the whole reply is one bare `tool_name({json})`
/// pseudo-call. Models under context pressure degrade to this shape (observed
/// live: a mid-task Qwen3 emitting `read_file({"path":"parts3.txt"})` as plain
/// text, which would otherwise end the loop with a tool call as the "answer").
/// Deliberately strict — the WHOLE trimmed text, one identifier, one balanced
/// JSON object — so prose that merely mentions a call never matches.
fn parse_bare_call(text: &str) -> Vec<ToolCall> {
    let t = text.trim();
    let Some(open) = t.find("({") else {
        return Vec::new();
    };
    let name = &t[..open];
    if name.is_empty()
        || !name
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
    {
        return Vec::new();
    }
    let Some(rest) = t.strip_suffix(')') else {
        return Vec::new();
    };
    let json_part = &rest[open + 1..];
    match serde_json::from_str::<Value>(json_part) {
        Ok(args @ Value::Object(_)) => vec![ToolCall {
            name: name.to_string(),
            args,
        }],
        _ => Vec::new(),
    }
}

/// Parse JSON leniently for model-emitted tool calls. On Windows, models often
/// place paths like `C:\workspace\docs` or `\\?\C:\x` inside JSON string values without
/// escaping the backslashes — invalid JSON. When a strict parse fails, repair any
/// backslash that does not begin a valid JSON escape by doubling it, then retry
/// once. Returns `None` if it still will not parse.
fn json_from_str_lenient(s: &str) -> Option<Value> {
    if let Ok(value) = serde_json::from_str::<Value>(s) {
        return Some(value);
    }
    // Exact live Qwen3 failure observed in Web Code:
    //
    //   "content:""import tkinter as tk\n..."
    //
    // The model put the key/value colon inside the closing key quote.  The
    // remainder of the write_file envelope (including its escaped source) was
    // valid, but dropping the whole call made the agent reprompt until its
    // no-progress guard fired.  Repair only a structural `content` key (after
    // `{` or `,`) and only after strict JSON has already failed; a literal copy
    // of the same bytes inside otherwise-valid file content is therefore never
    // rewritten.
    let repaired_key = repair_misquoted_content_key(s).unwrap_or_else(|| s.to_string());
    let repaired_paths = repair_path_backslashes(&repaired_key);
    if let Ok(value) = serde_json::from_str::<Value>(&repaired_paths) {
        return Some(value);
    }
    serde_json::from_str::<Value>(&repair_invalid_json_escapes(&repaired_paths)).ok()
}

fn repair_misquoted_content_key(s: &str) -> Option<String> {
    const MALFORMED_QUOTED: &str = "\"content:\"\"";
    const MALFORMED_PYTHON_WRAPPER: &str = "\"content:\"python -c '";
    const REPAIRED: &str = "\"content\":\"";

    // This recovery changes executable file bytes, so keep it scoped to the
    // only advertised tool whose schema owns `content`. Unknown envelopes and
    // malformed shell/network calls remain inert.
    if !(s.contains(r#""name":"write_file""#) || s.contains(r#""name": "write_file""#)) {
        return None;
    }
    let structural_start = |needle: &str| {
        let start = s.find(needle)?;
        s[..start]
            .chars()
            .rev()
            .find(|character| !character.is_whitespace())
            .is_some_and(|character| matches!(character, '{' | ','))
            .then_some(start)
    };

    if let Some(start) = structural_start(MALFORMED_QUOTED) {
        let mut output = s.to_string();
        output.replace_range(start..start + MALFORMED_QUOTED.len(), REPAIRED);
        return Some(output);
    }

    let start = structural_start(MALFORMED_PYTHON_WRAPPER)?;
    let source_start = start + MALFORMED_PYTHON_WRAPPER.len();
    let closing_relative = s[source_start..].rfind("'\"")?;
    let closing_apostrophe = source_start + closing_relative;
    // The shell-style quote must terminate the content value immediately
    // before the arguments/envelope braces. Do not peel quotes out of source.
    if !s[closing_apostrophe + 2..].trim_start().starts_with('}') {
        return None;
    }
    let mut output = String::with_capacity(s.len());
    output.push_str(&s[..start]);
    output.push_str(REPAIRED);
    output.push_str(&s[source_start..closing_apostrophe]);
    output.push_str(&s[closing_apostrophe + 1..]); // keep the JSON quote, drop only `'`
    Some(output)
}

/// Preserve model-authored source while repairing invalid JSON string escapes.
///
/// A frequent Qwen failure is embedding Python such as `it\'s` directly in a
/// JSON string. JSON does not define `\'`, even though the backslash is needed
/// by the Python source. Doubling only invalid escapes makes the JSON decode to
/// the exact intended `\'`. Valid JSON escapes (`\n`, `\t`, `\u1234`, …) are
/// deliberately untouched.
fn repair_invalid_json_escapes(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 8);
    let mut chars = s.chars().peekable();
    let mut in_string = false;
    while let Some(character) = chars.next() {
        if in_string && character == '\\' {
            // Consume the escape and its following char TOGETHER. A `\"` must not
            // fall through to the `"` arm below, or it would flip `in_string` and
            // desync the tracker — a command like `echo \"\$i.txt\"` then leaves
            // the later `\$` treated as outside a string and never repaired, so
            // the whole call stays unparseable. (This was the live failure: Qwen
            // over-escaping `$` inside a run_shell command killed the turn.)
            let next = chars.next();
            let valid = matches!(
                next,
                Some('"' | '\\' | '/' | 'b' | 'f' | 'n' | 'r' | 't' | 'u')
            );
            if !valid {
                out.push('\\');
            }
            out.push(character);
            if let Some(next) = next {
                out.push(next);
            }
            continue;
        }
        if character == '"' {
            in_string = !in_string;
        }
        out.push(character);
    }
    out
}

/// Public wrapper for the structured-`tool_calls` path: parse an arguments string
/// leniently, defaulting to an empty object.
pub(crate) fn json_args_lenient(s: &str) -> Value {
    json_from_str_lenient(s).unwrap_or_else(|| Value::Object(Default::default()))
}

/// Repair unescaped Windows separators only in path-shaped JSON fields. Other
/// strings may contain valid escapes such as `\n` or `\uXXXX`; rewriting the
/// entire arguments object would silently change file content and patterns.
fn repair_path_backslashes(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 8);
    let mut cursor = 0usize;
    let mut repair_next_string = false;
    while cursor < s.len() {
        let Some(relative_start) = s[cursor..].find('"') else {
            out.push_str(&s[cursor..]);
            break;
        };
        let start = cursor + relative_start;
        out.push_str(&s[cursor..start]);
        let Some(end) = json_string_end(s, start) else {
            out.push_str(&s[start..]);
            break;
        };
        let token = &s[start..=end];
        let next = s[end + 1..].trim_start().chars().next();
        if next == Some(':') {
            let key = serde_json::from_str::<String>(token).unwrap_or_default();
            repair_next_string = matches!(key.as_str(), "path" | "cwd");
            out.push_str(token);
        } else if repair_next_string {
            out.push_str(&repair_path_string(token));
            repair_next_string = false;
        } else {
            out.push_str(token);
        }
        cursor = end + 1;
    }
    out
}

fn json_string_end(s: &str, start: usize) -> Option<usize> {
    let mut escaped = false;
    for (offset, character) in s[start + 1..].char_indices() {
        if escaped {
            escaped = false;
        } else if character == '\\' {
            escaped = true;
        } else if character == '"' {
            return Some(start + 1 + offset);
        }
    }
    None
}

fn repair_path_string(token: &str) -> String {
    let mut out = String::with_capacity(token.len() + 4);
    let mut chars = token.chars().peekable();
    while let Some(character) = chars.next() {
        if character == '\\' {
            match chars.peek() {
                Some('"' | '\\') => {
                    out.push('\\');
                    out.push(chars.next().unwrap());
                }
                _ => out.push_str("\\\\"),
            }
        } else {
            out.push(character);
        }
    }
    out
}

/// `[TOOL_CALLS] [{"name": …, "arguments": {…}}, …]` (Mistral Instruct v0.3+).
fn parse_mistral(text: &str) -> Vec<ToolCall> {
    let marker = "[TOOL_CALLS]";
    if let Some(idx) = text.find(marker) {
        let rest = text[idx + marker.len()..].trim();
        if let Some(value) = json_from_str_lenient(rest) {
            return calls_from_value(&value);
        }
        // The model sometimes appends an EOS token or trailing text after the array;
        // try to extract the first balanced [...] substring.
        if let Some(start) = rest.find('[') {
            let slice = &rest[start..];
            if let Some(value) = json_from_str_lenient(slice) {
                return calls_from_value(&value);
            }
        }
    }
    // Mistral v0.3 GGUF emits bare JSON arrays without [TOOL_CALLS] marker.
    // Extract the first balanced [...] block, ignoring trailing prose.
    if let Some(arr_slice) = first_json_array(text.trim()) {
        if let Some(value) = json_from_str_lenient(arr_slice) {
            let calls = calls_from_value(&value);
            if !calls.is_empty() {
                return calls;
            }
        }
    }
    vec![]
}

/// `<tool_call>{ "name": …, "arguments": { … } }</tool_call>` blocks (Qwen/Hermes).
fn parse_hermes(text: &str) -> Vec<ToolCall> {
    let mut calls = Vec::new();
    let mut rest = text;
    while let Some(start) = rest.find("<tool_call>") {
        rest = &rest[start + "<tool_call>".len()..];
        let inner = match rest.find("</tool_call>") {
            Some(end) => {
                let inner = &rest[..end];
                rest = &rest[end + "</tool_call>".len()..];
                inner
            }
            None => rest,
        };
        if let Some(value) = json_from_str_lenient(inner.trim()) {
            if let Some(call) = call_from_obj(&value) {
                calls.push(call);
            }
        }
    }
    calls
}

/// Ornith / Qwen3.5 custom XML tool calls:
/// `<tool_call>\n<function=NAME>\n<parameter=ARG>\nVALUE\n</parameter>…\n</function>\n</tool_call>`.
/// Parses on the `<function=…>` boundary (the `<tool_call>` wrapper is optional in
/// practice), so a bare function block still lifts. Each `<parameter=ARG>` value keeps
/// the template's wrapper newline stripped; values that look like JSON objects/arrays
/// are decoded (the template `tojson`s mapping/sequence args), scalars stay strings.
fn parse_ornith(text: &str) -> Vec<ToolCall> {
    let mut calls = Vec::new();
    let mut rest = text;
    while let Some(fstart) = rest.find("<function=") {
        let after = &rest[fstart + "<function=".len()..];
        let Some(name_end) = after.find('>') else {
            break;
        };
        let name = after[..name_end].trim().to_string();
        let body = &after[name_end + 1..];
        let (params_blob, next) = match body.find("</function>") {
            Some(end) => (&body[..end], &body[end + "</function>".len()..]),
            None => (body, ""),
        };

        let mut args = serde_json::Map::new();
        let mut p = params_blob;
        while let Some(ps) = p.find("<parameter=") {
            let pa = &p[ps + "<parameter=".len()..];
            let Some(pname_end) = pa.find('>') else { break };
            let pname = pa[..pname_end].trim().to_string();
            let pbody = &pa[pname_end + 1..];
            let (pval, pnext) = match pbody.find("</parameter>") {
                Some(end) => (&pbody[..end], &pbody[end + "</parameter>".len()..]),
                None => (pbody, ""),
            };
            // The template wraps the value as `>\nVALUE\n</parameter>`; strip exactly
            // one leading + one trailing newline to recover VALUE verbatim.
            let v = pval.strip_prefix('\n').unwrap_or(pval);
            let v = v.strip_suffix('\n').unwrap_or(v);
            let trimmed = v.trim();
            let value = if trimmed.starts_with('{') || trimmed.starts_with('[') {
                serde_json::from_str::<Value>(trimmed)
                    .unwrap_or_else(|_| Value::String(v.to_string()))
            } else {
                Value::String(v.to_string())
            };
            if !pname.is_empty() {
                args.insert(pname, value);
            }
            p = pnext;
        }

        if !name.is_empty() {
            calls.push(ToolCall {
                name,
                args: Value::Object(args),
            });
        }
        rest = next;
    }
    calls
}

/// Gemma 4 `<|tool_call>call:NAME{argsâ€¦}<tool_call|>` envelopes, delegated to
/// the serve lane's certified parser (one scanner, no drift): both the native
/// pseudo-JSON argument dialect and the raw-JSON arm lift, and a truncated
/// envelope parses to its emitted prefix.
fn parse_gemma4(text: &str) -> Vec<ToolCall> {
    crate::api::parse_gemma4_tool_calls_json(text)
        .into_iter()
        .filter_map(|call| {
            let name = call["function"]["name"].as_str()?.to_string();
            let args = serde_json::from_str(call["function"]["arguments"].as_str()?).ok()?;
            Some(ToolCall { name, args })
        })
        .collect()
}

/// Bare/`python_tag`-wrapped JSON tool call(s) (Llama 3.x).
fn parse_json(text: &str) -> Vec<ToolCall> {
    let cleaned = strip_markers(text);
    let trimmed = cleaned.trim();
    if let Some(value) = json_from_str_lenient(trimmed) {
        return calls_from_value(&value);
    }
    // Otherwise recover every balanced object. Some certified Llama 3 rows
    // emit several native calls as `{...}; {...}; {...}` instead of a JSON
    // array. Executing only the first loses follow-up verification, while
    // treating the whole line as prose loses every call.
    let mut calls = Vec::new();
    let mut rest = trimmed;
    while let Some(slice) = first_json_object(rest) {
        let start = rest.find(slice).unwrap_or(0);
        if let Some(value) = json_from_str_lenient(slice) {
            calls.extend(calls_from_value(&value));
        }
        let consumed = start.saturating_add(slice.len());
        if consumed >= rest.len() {
            break;
        }
        rest = &rest[consumed..];
    }
    // A malformed write can share a turn with valid calls. Do not let the
    // successfully parsed tail hide the artifact-producing call: recover the
    // narrowly recognised write and retain the valid verification calls too.
    if !calls.iter().any(|call| call.name == "write_file") {
        if let Some(call) = recover_malformed_write_file(trimmed) {
            calls.insert(0, call);
        }
    }
    calls
}

/// Recover only a clearly delimited `write_file` envelope whose source contains
/// unescaped quotes. This is intentionally not a general malformed-JSON parser:
/// writes remain sandbox-validated and approval-gated, while malformed shell or
/// network calls stay inert. Llama 3 occasionally emits valid native envelope
/// structure but forgets to JSON-escape quotes inside the `content` value.
fn recover_malformed_write_file(text: &str) -> Option<ToolCall> {
    let trimmed = text.trim();
    if !trimmed.starts_with('{')
        || !(trimmed.contains(r#""name":"write_file""#)
            || trimmed.contains(r#""name": "write_file""#))
    {
        return None;
    }
    let content_key = trimmed.find(r#""content""#)?;
    let after_key = &trimmed[content_key + r#""content""#.len()..];
    let content_open = after_key.find('"')?;
    let encoded_content = &after_key[content_open + 1..];
    let (encoded_content, path) = if let Some((content, path_tail)) = encoded_content
        .rsplit_once(r#"", "path": ""#)
        .or_else(|| encoded_content.rsplit_once(r#"","path":""#))
    {
        let path_end = path_tail.find('"')?;
        let path = &path_tail[..path_end];
        if path.is_empty() || path.contains(['\n', '\r', '\0']) {
            return None;
        }
        (content, Some(decode_jsonish_string(path)))
    } else {
        // Preserve a recognisable malformed write even when the model omitted
        // its required path. The call remains non-executable: normal tool
        // validation will reject the missing path and feed that typed error
        // back to the model. Dropping it here would misclassify source as a
        // final answer and bypass the model's recovery turn entirely.
        let content = encoded_content
            .strip_suffix(r#""}}"#)
            .or_else(|| encoded_content.strip_suffix(r#""} }"#))
            // Exact Llama 3.2 3B live failure: it closes `parameters` but
            // omits the outer envelope brace after a long content value.
            .or_else(|| encoded_content.strip_suffix(r#""}"#))?;
        (content, None)
    };
    let mut args = serde_json::Map::new();
    args.insert(
        "content".into(),
        Value::String(decode_jsonish_string(encoded_content)),
    );
    if let Some(path) = path {
        args.insert("path".into(), Value::String(path));
    }
    Some(ToolCall {
        name: "write_file".into(),
        args: Value::Object(args),
    })
}

fn decode_jsonish_string(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    let mut chars = value.chars().peekable();
    while let Some(character) = chars.next() {
        if character != '\\' {
            out.push(character);
            continue;
        }
        match chars.next() {
            Some('n') => out.push('\n'),
            Some('r') => out.push('\r'),
            Some('t') => out.push('\t'),
            Some('"') => out.push('"'),
            Some('\\') => out.push('\\'),
            Some(other) => {
                out.push('\\');
                out.push(other);
            }
            None => out.push('\\'),
        }
    }
    out
}

fn calls_from_value(value: &Value) -> Vec<ToolCall> {
    match value {
        Value::Array(items) => items.iter().filter_map(call_from_obj).collect(),
        Value::Object(_) => call_from_obj(value).into_iter().collect(),
        _ => Vec::new(),
    }
}

/// Build a call from an object: `name` + args from `parameters`/`arguments`/the
/// object minus the envelope keys. Returns None if there's no usable name.
fn call_from_obj(value: &Value) -> Option<ToolCall> {
    let obj = value.as_object()?;
    // Some models nest under "function": {"name":…,"arguments":…}.
    if let Some(func) = obj.get("function").and_then(Value::as_object) {
        let name = func.get("name").and_then(Value::as_str)?.to_string();
        let args = func
            .get("arguments")
            .or_else(|| func.get("parameters"))
            .cloned()
            .map(coerce_args)
            .unwrap_or_else(|| Value::Object(Default::default()));
        return Some(ToolCall { name, args });
    }
    let name = obj.get("name").and_then(Value::as_str)?.to_string();
    let args = obj
        .get("parameters")
        .or_else(|| obj.get("arguments"))
        .cloned()
        .map(coerce_args)
        .unwrap_or_else(|| {
            let mut rest = obj.clone();
            rest.remove("name");
            rest.remove("type");
            Value::Object(rest)
        });
    Some(ToolCall { name, args })
}

/// Arguments are sometimes a JSON *string* — decode it to an object when so.
fn coerce_args(value: Value) -> Value {
    if let Value::String(s) = &value {
        if let Ok(parsed) = serde_json::from_str::<Value>(s) {
            return parsed;
        }
    }
    value
}

fn strip_markers(text: &str) -> String {
    let mut s = text.to_string();
    for marker in [
        "<|python_tag|>",
        "<|eom_id|>",
        "<|eot_id|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "```json",
        "```",
    ] {
        s = s.replace(marker, " ");
    }
    s
}

/// First balanced `{…}` substring (depth-aware, ignores braces in strings).
fn first_json_object(s: &str) -> Option<&str> {
    let bytes = s.as_bytes();
    let start = bytes.iter().position(|&b| b == b'{')?;
    let mut depth = 0usize;
    let mut in_str = false;
    let mut escaped = false;
    for (i, &b) in bytes.iter().enumerate().skip(start) {
        if in_str {
            if escaped {
                escaped = false;
            } else if b == b'\\' {
                escaped = true;
            } else if b == b'"' {
                in_str = false;
            }
            continue;
        }
        match b {
            b'"' => in_str = true,
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(&s[start..=i]);
                }
            }
            _ => {}
        }
    }
    None
}

/// First balanced `[…]` substring (depth-aware, ignores brackets in strings).
fn first_json_array(s: &str) -> Option<&str> {
    let bytes = s.as_bytes();
    let start = bytes.iter().position(|&b| b == b'[')?;
    let mut depth = 0usize;
    let mut in_str = false;
    let mut escaped = false;
    for (i, &b) in bytes.iter().enumerate().skip(start) {
        if in_str {
            if escaped {
                escaped = false;
            } else if b == b'\\' {
                escaped = true;
            } else if b == b'"' {
                in_str = false;
            }
            continue;
        }
        match b {
            b'"' => in_str = true,
            b'[' => depth += 1,
            b']' => {
                depth -= 1;
                if depth == 0 {
                    return Some(&s[start..=i]);
                }
            }
            _ => {}
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_call_parser_corpus_accepts_unambiguous_variants_and_rejects_truncation() {
        let accepted = [
            (
                "qwen3",
                r#"<tool_call>{"name":"read_file","arguments":{"path":"src/lib.rs"}}</tool_call>"#,
                "read_file",
                "src/lib.rs",
            ),
            // A complete JSON object is still unambiguous when a small model
            // omits only the decorative closing tag.
            (
                "qwen3",
                r#"<tool_call>{"name":"read_file","arguments":{"path":"src/lib.rs"}}"#,
                "read_file",
                "src/lib.rs",
            ),
            (
                "mistral",
                r#"[TOOL_CALLS] [{"name":"read_file","arguments":{"path":"src/lib.rs"}}]"#,
                "read_file",
                "src/lib.rs",
            ),
            (
                "ornith",
                "<tool_call>\n<function=read_file>\n<parameter=path>\nsrc/lib.rs\n</parameter>\n</function>\n</tool_call>",
                "read_file",
                "src/lib.rs",
            ),
            (
                "qwen3",
                r#"read_file({"path":"src/lib.rs"})"#,
                "read_file",
                "src/lib.rs",
            ),
        ];
        for (family, text, expected_name, expected_path) in accepted {
            let calls = parse(text, family);
            assert_eq!(calls.len(), 1, "family={family}, text={text}");
            assert_eq!(calls[0].name, expected_name);
            assert_eq!(calls[0].args["path"], expected_path);
        }

        for truncated_or_ambiguous in [
            r#"<tool_call>{"name":"read_file","arguments":{"path":"src/lib.rs""#,
            r#"<tool_call>{"arguments":{"path":"src/lib.rs"}}</tool_call>"#,
            r#"I might call read_file({"path":"src/lib.rs"}) after checking."#,
            r#"<tool_call><function=read_file><parameter=path>src/lib.rs"#,
        ] {
            assert!(
                parse(truncated_or_ambiguous, "qwen3").is_empty(),
                "must not fabricate a call from {truncated_or_ambiguous}"
            );
        }
    }

    /// The live failure that killed a Code turn: Qwen over-escapes `$` inside a
    /// run_shell command, and the command also contains escaped quotes `\"`. The
    /// `\"` used to desync the repair's in-string tracker so the later `\$` was
    /// never doubled, leaving the call unparseable and looping the turn to death.
    #[test]
    fn hermes_call_with_escaped_quotes_and_escaped_dollar_is_recovered() {
        let text = r#"<tool_call>
{"name": "run_shell", "arguments": {"command": "seq 1 100 | while read -r i; do echo \"\$i.txt\"; done"}}
</tool_call>"#;
        let calls = parse(text, "qwen3");
        assert_eq!(calls.len(), 1, "the call must be recovered, got {calls:?}");
        assert_eq!(calls[0].name, "run_shell");
        let command = calls[0].args["command"].as_str().unwrap();
        assert!(
            command.contains("seq 1 100") && command.contains("i.txt"),
            "command survived repair: {command}"
        );
    }

    #[test]
    fn parses_llama_json_with_parameters() {
        let out = parse(
            "<|python_tag|>{\"name\": \"read_file\", \"parameters\": {\"path\": \"src/main.rs\"}}<|eom_id|>",
            "llama_bpe_decoder",
        );
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].name, "read_file");
        assert_eq!(out[0].args["path"], "src/main.rs");
    }

    #[test]
    fn parses_hermes_qwen_tool_call_tags() {
        let out = parse(
            "sure<tool_call>{\"name\": \"list_dir\", \"arguments\": {\"path\": \".\"}}</tool_call>",
            "qwen3",
        );
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].name, "list_dir");
        assert_eq!(out[0].args["path"], ".");
    }

    #[test]
    fn parses_windows_path_with_unescaped_backslashes() {
        // Qwen echoes a Windows workspace path with single (JSON-invalid) backslashes.
        let out = parse(
            r#"<tool_call>{"name": "list_dir", "arguments": {"path": "C:\workspace\docs"}}</tool_call>"#,
            "qwen3",
        );
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].name, "list_dir");
        assert_eq!(out[0].args["path"], r"C:\workspace\docs");
    }

    #[test]
    fn lenient_parse_preserves_valid_escapes() {
        // Valid JSON (with legitimate \n and \") must parse strictly and be untouched.
        let out = parse(
            r#"<tool_call>{"name":"write_file","arguments":{"path":"a.txt","content":"line1\nline2 \"q\""}}</tool_call>"#,
            "qwen3",
        );
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].args["content"], "line1\nline2 \"q\"");
    }

    #[test]
    fn repairs_live_qwen_misquoted_write_content_key() {
        // Exact structural corruption captured from Qwen3-4B-Q4_K_M in Web
        // Code. The path and source are valid; only the colon terminating the
        // known `content` key moved one byte to the left.
        let out = parse(
            r#"<tool_call>
{"name":"write_file","arguments":{"path":"tic_tac_toe.py","content:""import tkinter as tk\n\nroot = tk.Tk()\nroot.title(\"Tic Tac Toe\")\n"}}}
</tool_call>"#,
            "qwen3",
        );
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].name, "write_file");
        assert_eq!(out[0].args["path"], "tic_tac_toe.py");
        assert_eq!(
            out[0].args["content"],
            "import tkinter as tk\n\nroot = tk.Tk()\nroot.title(\"Tic Tac Toe\")\n"
        );
    }

    #[test]
    fn valid_content_containing_misquoted_key_bytes_is_not_rewritten() {
        let out = parse(
            r#"<tool_call>{"name":"write_file","arguments":{"path":"note.txt","content":"literal \"content:\"\" marker"}}</tool_call>"#,
            "qwen3",
        );
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].args["content"], "literal \"content:\"\" marker");
    }

    #[test]
    fn repairs_live_qwen_misquoted_python_wrapper_write() {
        // Second exact live variation: Qwen placed a shell-style python -c
        // wrapper where the write_file content string should begin.
        let out = parse(
            r#"<tool_call>
{"name":"write_file","arguments":{"path":"tic_tac_toe.py","content:"python -c 'import tkinter as tk\nroot = tk.Tk()\nroot.title(\"Tic Tac Toe\")'"}}
</tool_call>"#,
            "qwen3",
        );
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].name, "write_file");
        assert_eq!(out[0].args["path"], "tic_tac_toe.py");
        assert_eq!(
            out[0].args["content"],
            "import tkinter as tk\nroot = tk.Tk()\nroot.title(\"Tic Tac Toe\")"
        );
    }

    #[test]
    fn python_wrapper_repair_is_write_file_only() {
        assert!(parse(
            r#"<tool_call>{"name":"run_shell","arguments":{"content:"python -c 'print(1)'"}}</tool_call>"#,
            "qwen3"
        )
        .is_empty());
    }

    #[test]
    fn lenient_path_repair_does_not_corrupt_other_string_escapes() {
        let out = parse(
            r#"<tool_call>{"name":"write_file","arguments":{"path":"C:\workspace\note.txt","content":"line1\nline2\t\u263A"}}</tool_call>"#,
            "qwen3",
        );
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].args["path"], r"C:\workspace\note.txt");
        assert_eq!(out[0].args["content"], "line1\nline2\t☺");
    }

    #[test]
    fn parses_semicolon_separated_llama_json_calls() {
        let out = parse(
            r#"{"name":"write_file","parameters":{"path":"game.py","content":"print('x')\n"}}; {"name":"run_shell","parameters":{"command":"py -m py_compile game.py"}}; {"name":"read_file","parameters":{"path":"game.py"}}"#,
            "llama_bpe_decoder",
        );
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].name, "write_file");
        assert_eq!(out[1].name, "run_shell");
        assert_eq!(out[2].name, "read_file");
        assert_eq!(out[0].args["path"], "game.py");
    }

    #[test]
    fn recovers_unescaped_source_quotes_only_for_write_file() {
        let text = r#"{"name": "write_file", "parameters": {"content": "self.window.title("Tic Tac Toe)\nprint("hi")", "path": "tic_tac_toe.py"}}"#;
        let out = parse(text, "llama_bpe_decoder");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].name, "write_file");
        assert_eq!(out[0].args["path"], "tic_tac_toe.py");
        assert_eq!(
            out[0].args["content"],
            "self.window.title(\"Tic Tac Toe)\nprint(\"hi\")"
        );
        assert!(parse(
            r#"{"name":"run_shell","parameters":{"command":"echo "oops""}}"#,
            "llama_bpe_decoder"
        )
        .is_empty());
    }

    #[test]
    fn malformed_write_is_not_lost_when_later_llama_calls_are_valid() {
        let text = concat!(
            r#"{"name": "write_file", "parameters": {"content": "root.title("Tic Tac Toe")\n", "path": "tic_tac_toe.py"}}"#,
            r#"; {"name":"run_shell","parameters":{"command":"py tic_tac_toe.py"}}"#,
            r#"; {"name":"search","parameters":{"pattern":"TicTacToe","path":"."}}"#,
        );
        let out = parse(text, "llama_bpe_decoder");
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].name, "write_file");
        assert_eq!(out[0].args["path"], "tic_tac_toe.py");
        assert_eq!(out[1].name, "run_shell");
        assert_eq!(out[2].name, "search");
    }

    #[test]
    fn malformed_write_without_path_surfaces_for_typed_validation_error() {
        // Exact live shape has only one trailing brace: the model closes the
        // parameters object but omits the outer envelope brace.
        let text = r#"{"name": "write_file", "parameters": {"content": "root.title("Tic Tac Toe)\ngame.run()"}"#;
        let out = parse(text, "llama_bpe_decoder");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].name, "write_file");
        assert_eq!(
            out[0].args["content"],
            "root.title(\"Tic Tac Toe)\ngame.run()"
        );
        assert!(out[0].args.get("path").is_none());
    }

    #[test]
    fn lenient_parse_preserves_backslash_apostrophe_in_embedded_source() {
        let out = parse(
            r#"<tool_call>{"name":"write_file","arguments":{"path":"game.py","content":"print('it\'s your turn')\n"}}</tool_call>"#,
            "qwen3",
        );
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].name, "write_file");
        assert_eq!(out[0].args["content"], "print('it\\'s your turn')\n");
    }

    #[test]
    fn json_args_lenient_repairs_or_defaults() {
        assert_eq!(json_args_lenient(r#"{"path":"C:\a\b"}"#)["path"], r"C:\a\b");
        assert_eq!(
            json_args_lenient("not json"),
            Value::Object(Default::default())
        );
    }

    #[test]
    fn parses_call_embedded_in_prose() {
        let out = parse(
            "I will read it. {\"name\":\"read_file\",\"parameters\":{\"path\":\"a\"}} done",
            "llama",
        );
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].name, "read_file");
    }

    #[test]
    fn bare_pseudo_call_is_recognised_for_every_family() {
        for family in ["qwen", "llama", "hermes"] {
            let calls = parse(r#"read_file({"path":"parts3.txt"})"#, family);
            assert_eq!(calls.len(), 1, "family {family}");
            assert_eq!(calls[0].name, "read_file");
            assert_eq!(calls[0].args["path"], "parts3.txt");
        }
    }

    #[test]
    fn prose_mentioning_a_call_is_not_a_bare_call() {
        for text in [
            r#"I will now run read_file({"path":"a"}) to check."#,
            r#"The answer is 42 (see notes)."#,
            r#"read_file(not json)"#,
            r#"Read_File({"path":"a"})"#,
        ] {
            assert!(
                parse(text, "qwen").is_empty(),
                "{text:?} must not parse as a call"
            );
        }
    }

    #[test]
    fn plain_answer_yields_no_calls() {
        assert!(parse("The file has 3 lines.", "llama").is_empty());
    }

    #[test]
    fn malformed_json_is_clean_not_a_panic() {
        // Looks like a call but is broken JSON → no calls, no panic.
        assert!(parse("{\"name\": \"read_file\", \"parameters\": {bad", "llama").is_empty());
        assert!(parse("<tool_call>{not json}</tool_call>", "qwen").is_empty());
        // Truncated mid-string and empty input.
        assert!(parse(
            "{\"name\":\"read_file\",\"parameters\":{\"path\":\"no",
            "llama"
        )
        .is_empty());
        assert!(parse("", "llama").is_empty());
    }

    #[test]
    fn double_encoded_args_string_is_normalized_to_object() {
        // Some models emit `parameters`/`arguments` as a JSON-encoded *string*.
        let out = parse(
            "{\"name\":\"read_file\",\"arguments\":\"{\\\"path\\\":\\\"a.txt\\\"}\"}",
            "llama",
        );
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].args["path"], "a.txt"); // normalized to a real object
    }

    #[test]
    fn function_envelope_is_unwrapped() {
        // OpenAI-shaped output the model sometimes mirrors back.
        let out = parse(
            "{\"type\":\"function\",\"function\":{\"name\":\"list_dir\",\"arguments\":{\"path\":\".\"}}}",
            "llama",
        );
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].name, "list_dir");
        assert_eq!(out[0].args["path"], ".");
    }

    #[test]
    fn multiple_calls_in_one_turn() {
        // Hermes: two tagged calls.
        let hermes = parse(
            "<tool_call>{\"name\":\"read_file\",\"arguments\":{\"path\":\"a\"}}</tool_call>\
             <tool_call>{\"name\":\"list_dir\",\"arguments\":{\"path\":\".\"}}</tool_call>",
            "qwen3",
        );
        assert_eq!(hermes.len(), 2);
        assert_eq!(hermes[0].name, "read_file");
        assert_eq!(hermes[1].name, "list_dir");
        // Llama: a JSON array of calls.
        let arr = parse(
            "[{\"name\":\"read_file\",\"parameters\":{\"path\":\"a\"}},{\"name\":\"search\",\"parameters\":{\"pattern\":\"x\"}}]",
            "llama",
        );
        assert_eq!(arr.len(), 2);
        assert_eq!(arr[1].name, "search");
    }

    #[test]
    fn trailing_and_leading_prose_around_call() {
        let out = parse(
            "Sure, I'll read it now:\n<tool_call>{\"name\":\"read_file\",\"arguments\":{\"path\":\"a\"}}</tool_call>\nDone.",
            "qwen",
        );
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].name, "read_file");
    }

    #[test]
    fn schema_echo_parses_to_name_with_wrong_args_for_the_gate_to_reject() {
        // The exact 1B failure mode: name is right, args are the schema. The
        // parser must surface it (name parsed) so validate() rejects it with a
        // typed error rather than the parser silently "succeeding".
        let out = parse(
            "{\"name\":\"read_file\",\"parameters\":{\"properties\":{\"path\":{\"type\":\"string\"}},\"required\":[\"path\"],\"type\":\"object\"}}",
            "llama",
        );
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].name, "read_file");
        assert!(out[0].args.get("path").is_none()); // no real value → gate rejects
    }

    #[test]
    fn parses_mistral_tool_calls_marker() {
        let out = parse(
            "[TOOL_CALLS] [{\"name\": \"read_file\", \"arguments\": {\"path\": \"notes.txt\"}}]",
            "mistral",
        );
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].name, "read_file");
        assert_eq!(out[0].args["path"], "notes.txt");
    }

    #[test]
    fn parses_mistral_multiple_tool_calls() {
        let out = parse(
            "[TOOL_CALLS] [{\"name\": \"read_file\", \"arguments\": {\"path\": \"a.txt\"}}, {\"name\": \"list_dir\", \"arguments\": {\"path\": \".\"}}]",
            "mistral",
        );
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].name, "read_file");
        assert_eq!(out[0].args["path"], "a.txt");
        assert_eq!(out[1].name, "list_dir");
        assert_eq!(out[1].args["path"], ".");
    }

    #[test]
    fn mistral_falls_back_to_json_without_marker() {
        // If Mistral emits bare JSON (unlikely but possible), the fallback works.
        let out = parse(
            "{\"name\": \"read_file\", \"arguments\": {\"path\": \"x\"}}",
            "mistral",
        );
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].name, "read_file");
    }

    #[test]
    fn mistral_plain_answer_yields_no_calls() {
        assert!(parse("The file contains 3 lines of text.", "mistral").is_empty());
    }

    #[test]
    fn mistral_parses_bare_array_without_marker() {
        let out = parse(
            " [{\"name\": \"read_file\", \"arguments\": {\"path\": \"notes.txt\"}}]\n\nLet me read it.",
            "mistral",
        );
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].name, "read_file");
        assert_eq!(out[0].args["path"], "notes.txt");
    }

    #[test]
    fn mistral_parses_bare_multi_call_array() {
        let out = parse(
            "[{\"name\":\"read_file\",\"arguments\":{\"path\":\"a\"}},{\"name\":\"list_dir\",\"arguments\":{\"path\":\".\"}}]\nDone.",
            "mistral",
        );
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].name, "read_file");
        assert_eq!(out[1].name, "list_dir");
    }

    // ---- Ornith / qwen35 custom XML tool-call lift (Bug-1 gate) ----

    /// The exact bytes the Ornith chat template emits for a tool call, routed by the
    /// `qwen35` family (note: "qwen35" contains "qwen", so order matters).
    #[test]
    fn parses_ornith_single_tool_call() {
        let text = "<tool_call>\n<function=read_file>\n<parameter=path>\nnotes.txt\n</parameter>\n</function>\n</tool_call>";
        let out = parse(text, "qwen35");
        assert_eq!(out.len(), 1, "exactly one call, single parse");
        assert_eq!(out[0].name, "read_file");
        assert_eq!(out[0].args["path"], "notes.txt");
    }

    /// Reasoning must NOT contaminate the tool lift, and a natural-language preamble
    /// before the call (allowed by the template) is ignored.
    #[test]
    fn parses_ornith_call_after_think_and_preamble() {
        let text = "<think>\nI should read the file to count lines.\n</think>\n\nI'll read it now.\n<tool_call>\n<function=read_file>\n<parameter=path>\nnotes.txt\n</parameter>\n</function>\n</tool_call>";
        let out = parse(text, "qwen35");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].name, "read_file");
        assert_eq!(out[0].args["path"], "notes.txt");
    }

    /// Multiple parameters; a JSON-object-valued parameter is decoded, a scalar stays
    /// a string. No double-parse.
    #[test]
    fn parses_ornith_multi_param_and_json_value() {
        let text = "<tool_call>\n<function=edit_file>\n<parameter=path>\nsrc/x.rs\n</parameter>\n<parameter=edits>\n{\"a\": 1}\n</parameter>\n</function>\n</tool_call>";
        let out = parse(text, "qwen35");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].name, "edit_file");
        assert_eq!(out[0].args["path"], "src/x.rs");
        assert_eq!(out[0].args["edits"]["a"], 1);
    }

    /// Two calls in one message lift to two structured calls.
    #[test]
    fn parses_ornith_two_calls() {
        let text = "<tool_call>\n<function=read_file>\n<parameter=path>\na.txt\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=list_dir>\n<parameter=path>\n.\n</parameter>\n</function>\n</tool_call>";
        let out = parse(text, "qwen35");
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].name, "read_file");
        assert_eq!(out[0].args["path"], "a.txt");
        assert_eq!(out[1].name, "list_dir");
        assert_eq!(out[1].args["path"], ".");
    }

    /// Plain assistant text (no call) yields no calls — the loop treats it as a final
    /// answer rather than mis-firing a tool.
    #[test]
    fn ornith_plain_answer_no_calls() {
        let text = "<think>\nThe answer is 3.\n</think>\n\nThe file has 3 lines.";
        assert!(parse(text, "qwen35").is_empty());
    }

    /// Gemma 4 streamed content: envelope markers survive detokenization but
    /// the `<|"|>` string-quote token does not — the degraded bare-value scan
    /// still lifts the call.
    #[test]
    fn parses_gemma4_envelope_with_and_without_quote_token() {
        let quoted =
            "<|tool_call>call:read_file{path:<|\"|>notes.txt<|\"|>,start_line:2}<tool_call|>";
        let out = parse(quoted, "gemma4_a4b_moe_decoder");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].name, "read_file");
        assert_eq!(out[0].args["path"], "notes.txt");
        assert_eq!(out[0].args["start_line"], 2);

        let stripped = "<|tool_call>call:list_dir{path:.}<tool_call|>";
        let out = parse(stripped, "gemma4_a4b_moe_decoder");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].name, "list_dir");
        assert_eq!(out[0].args["path"], ".");
    }

    /// Gemma-family plain answers never mis-fire, and the JSON fallbacks stay
    /// reachable for gemma sessions.
    #[test]
    fn gemma4_plain_answer_no_calls_and_json_fallback_reachable() {
        assert!(parse("The file has 3 lines.", "gemma4_a4b_moe_decoder").is_empty());
        let out = parse(
            "{\"name\":\"read_file\",\"parameters\":{\"path\":\"a.txt\"}}",
            "gemma4_a4b_moe_decoder",
        );
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].name, "read_file");
    }
}
