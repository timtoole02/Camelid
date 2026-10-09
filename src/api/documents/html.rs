//! Reads an HTML page as the text a reader sees: no tags, scripts, styles or
//! comments, entities decoded, whitespace collapsed outside `<pre>`, and one
//! blank line between blocks. Headings come out as Markdown headings after a
//! blank line, so the library's chunker starts a new section at each one.
//!
//! Only html5ever's tokenizer is used; there is no tree to build, because the
//! text is all the library keeps. Bytes are read as UTF-8, like every other
//! text type the library reads.

use std::cell::RefCell;

use html5ever::tendril::StrTendril;
use html5ever::tokenizer::states::RawKind;
use html5ever::tokenizer::{
    BufferQueue, Tag, TagKind, Token, TokenSink, TokenSinkResult, Tokenizer, TokenizerOpts,
};
use html5ever::TokenizerResult;

pub(super) fn extract_text(bytes: &[u8]) -> String {
    let source = String::from_utf8_lossy(bytes);
    let source = source.strip_prefix('\u{feff}').unwrap_or(&source);
    let input = BufferQueue::default();
    input.push_back(StrTendril::from_slice(source));
    let tokenizer = Tokenizer::new(Reader::default(), TokenizerOpts::default());
    while let TokenizerResult::Script(()) = tokenizer.feed(&input) {}
    tokenizer.end();
    tokenizer.sink.state.into_inner().finish()
}

#[derive(Default)]
struct Reader {
    state: RefCell<Text>,
}

/// Elements whose content is never shown as text. Each is read raw by the
/// tokenizer, so markup inside it cannot leak out as text.
fn raw_kind(name: &str) -> Option<RawKind> {
    match name {
        "script" => Some(RawKind::ScriptData),
        "style" | "xmp" | "iframe" | "noembed" | "noframes" | "noscript" => Some(RawKind::Rawtext),
        "textarea" => Some(RawKind::Rcdata),
        _ => None,
    }
}

/// Elements whose content, though parsed as markup, is not page text. `head`
/// is not one: its end tag is optional, and everything in it that holds text is
/// either the title or read raw.
fn is_hidden_container(name: &str) -> bool {
    matches!(name, "template" | "svg" | "math" | "select" | "datalist")
}

fn heading_level(name: &str) -> Option<usize> {
    match name {
        "h1" => Some(1),
        "h2" => Some(2),
        "h3" => Some(3),
        "h4" => Some(4),
        "h5" => Some(5),
        "h6" => Some(6),
        _ => None,
    }
}

fn is_block(name: &str) -> bool {
    matches!(
        name,
        "address"
            | "article"
            | "aside"
            | "blockquote"
            | "body"
            | "caption"
            | "center"
            | "details"
            | "dialog"
            | "dir"
            | "div"
            | "dl"
            | "fieldset"
            | "figcaption"
            | "figure"
            | "footer"
            | "form"
            | "header"
            | "hgroup"
            | "hr"
            | "legend"
            | "main"
            | "menu"
            | "nav"
            | "ol"
            | "p"
            | "pre"
            | "section"
            | "summary"
            | "table"
            | "ul"
    )
}

fn is_line(name: &str) -> bool {
    matches!(name, "br" | "dd" | "dt" | "li" | "tr")
}

#[derive(Default)]
struct Text {
    out: String,
    pending_space: bool,
    /// The raw-text element being skipped, until its end tag.
    raw: Option<String>,
    /// The hidden container being skipped and how deeply it is nested.
    hidden: Option<(String, usize)>,
    pre_depth: usize,
    /// Where the open heading's marker starts, so an empty heading can be
    /// taken back out.
    heading: Option<(usize, usize)>,
    in_title: bool,
    title: String,
}

impl Text {
    fn at_line_start(&self) -> bool {
        self.out.is_empty() || self.out.ends_with('\n')
    }

    fn block_break(&mut self) {
        self.pending_space = false;
        if self.heading.is_some() {
            self.pending_space = true;
            return;
        }
        if self.out.is_empty() || self.out.ends_with("\n\n") {
            return;
        }
        self.out.push_str(if self.out.ends_with('\n') {
            "\n"
        } else {
            "\n\n"
        });
    }

    fn line_break(&mut self) {
        self.pending_space = false;
        if self.heading.is_some() {
            self.pending_space = true;
            return;
        }
        if !self.out.is_empty() {
            self.out.push('\n');
        }
    }

    fn push_text(&mut self, text: &str) {
        if self.pre_depth > 0 {
            self.pending_space = false;
            self.out
                .push_str(&text.replace("\r\n", "\n").replace('\r', "\n"));
            return;
        }
        for ch in text.chars() {
            if ch.is_whitespace() {
                self.pending_space = true;
                continue;
            }
            if self.pending_space
                && !self.out.ends_with(char::is_whitespace)
                && !self.out.is_empty()
            {
                self.out.push(' ');
            }
            self.pending_space = false;
            self.out.push(ch);
        }
    }

    fn start_tag(&mut self, tag: &Tag) -> TokenSinkResult<()> {
        let name: &str = &tag.name;
        // Raw elements switch the tokenizer even inside hidden content, so
        // markup in a script cannot close the element around it.
        if let Some(kind) = raw_kind(name) {
            if !tag.self_closing {
                self.raw = Some(name.to_string());
                return TokenSinkResult::RawData(kind);
            }
            return TokenSinkResult::Continue;
        }
        if let Some((hidden, depth)) = self.hidden.as_mut() {
            if hidden == name && !tag.self_closing {
                *depth += 1;
            }
            return TokenSinkResult::Continue;
        }
        // An SVG `<title>` is a tooltip, and is hidden above; this is the page's.
        if name == "title" {
            if !tag.self_closing {
                self.in_title = true;
                return TokenSinkResult::RawData(RawKind::Rcdata);
            }
            return TokenSinkResult::Continue;
        }
        if is_hidden_container(name) {
            if !tag.self_closing {
                self.hidden = Some((name.to_string(), 1));
            }
            return TokenSinkResult::Continue;
        }
        if name == "plaintext" {
            self.block_break();
            self.pre_depth += 1;
            return TokenSinkResult::Plaintext;
        }
        if let Some(level) = heading_level(name) {
            if self.heading.is_none() {
                self.block_break();
                let start = self.out.len();
                self.out.push_str(&"#".repeat(level));
                self.out.push(' ');
                self.heading = Some((start, self.out.len()));
            }
            return TokenSinkResult::Continue;
        }
        match name {
            "li" => {
                if !self.at_line_start() {
                    self.line_break();
                }
                if self.heading.is_none() {
                    self.out.push_str("- ");
                }
            }
            "td" | "th" if !self.at_line_start() && self.heading.is_none() => {
                self.pending_space = false;
                self.out.push_str(" | ");
            }
            "br" => self.line_break(),
            "pre" => {
                self.block_break();
                self.pre_depth += 1;
            }
            _ if is_line(name) && !self.at_line_start() => self.line_break(),
            _ if is_block(name) => self.block_break(),
            _ => {}
        }
        TokenSinkResult::Continue
    }

    fn end_tag(&mut self, tag: &Tag) {
        let name: &str = &tag.name;
        if name == "title" && self.in_title {
            self.in_title = false;
            return;
        }
        if let Some((hidden, depth)) = self.hidden.as_mut() {
            if hidden == name {
                *depth -= 1;
                if *depth == 0 {
                    self.hidden = None;
                }
            }
            return;
        }
        if heading_level(name).is_some() {
            if let Some((start, marker_end)) = self.heading.take() {
                if self.out.len() == marker_end {
                    self.out.truncate(start);
                } else {
                    self.block_break();
                }
            }
            return;
        }
        if name == "pre" {
            self.pre_depth = self.pre_depth.saturating_sub(1);
            self.block_break();
            return;
        }
        if is_line(name) && name != "br" {
            if !self.at_line_start() {
                self.line_break();
            }
        } else if is_block(name) {
            self.block_break();
        }
    }

    /// Trims every line's trailing space and keeps at most one blank line
    /// between paragraphs. The page's title leads, as a heading, unless the
    /// page already opens with it.
    fn finish(self) -> String {
        let mut body = String::new();
        let mut blank = false;
        for line in self.out.lines() {
            let line = line.trim_end();
            if line.is_empty() {
                blank = !body.is_empty();
                continue;
            }
            if !body.is_empty() {
                body.push_str(if blank { "\n\n" } else { "\n" });
            }
            blank = false;
            body.push_str(line);
        }

        let title = self.title.split_whitespace().collect::<Vec<_>>().join(" ");
        let first_line = body
            .lines()
            .next()
            .unwrap_or("")
            .trim_start_matches('#')
            .trim();
        if title.is_empty() || first_line == title {
            return body;
        }
        if body.is_empty() {
            return format!("# {title}");
        }
        format!("# {title}\n\n{body}")
    }
}

impl TokenSink for Reader {
    type Handle = ();

    fn process_token(&self, token: Token, _line_number: u64) -> TokenSinkResult<()> {
        let mut text = self.state.borrow_mut();
        match token {
            Token::TagToken(tag) => {
                if let Some(raw) = text.raw.as_deref() {
                    if tag.kind == TagKind::EndTag && raw == &*tag.name {
                        text.raw = None;
                    }
                    return TokenSinkResult::Continue;
                }
                match tag.kind {
                    TagKind::StartTag => return text.start_tag(&tag),
                    TagKind::EndTag => text.end_tag(&tag),
                }
            }
            Token::CharacterTokens(chars) => {
                if text.in_title {
                    text.title.push_str(&chars);
                } else if text.raw.is_none() && text.hidden.is_none() {
                    text.push_text(&chars);
                }
            }
            _ => {}
        }
        TokenSinkResult::Continue
    }
}

#[cfg(test)]
mod tests {
    use super::extract_text;

    #[test]
    fn reads_page_text_without_markup_scripts_or_styles() {
        let html = br#"<!DOCTYPE html>
<html><head><title>Refund policy</title>
<meta charset="utf-8"><style>p { color: red; }</style>
<script>if (a < b) { document.write("<p>injected</p>"); }</script>
</head>
<body>
  <!-- a comment -->
  <nav><a href="/">Home</a></nav>
  <h1>Refund   policy</h1>
  <p>Refunds take <b>five</b> business&nbsp;days &amp; are
     paid to the original card.</p>
  <noscript><p>Enable JavaScript</p></noscript>
  <template><p>template text</p></template>
  <svg><text>svg label</text></svg>
</body></html>"#;
        let text = extract_text(html);
        // The title leads because the page opens with its navigation, not with it.
        assert_eq!(
            text,
            "# Refund policy\n\nHome\n\n# Refund policy\n\n\
             Refunds take five business days & are paid to the original card."
        );
        for leaked in [
            "color",
            "injected",
            "document.write",
            "comment",
            "Enable",
            "template",
            "svg",
        ] {
            assert!(!text.contains(leaked), "{leaked} leaked into {text:?}");
        }
    }

    #[test]
    fn headings_start_sections_the_chunker_recognizes() {
        let html = b"<p>Intro text.</p><h2>Data <em>export</em></h2><p>Export within 30 days.</p>\
            <h3>  </h3><h3>Security<br>incidents</h3><p>We notify within 72 hours.</p>";
        assert_eq!(
            extract_text(html),
            "Intro text.\n\n## Data export\n\nExport within 30 days.\n\n### Security incidents\n\nWe notify within 72 hours."
        );
    }

    #[test]
    fn lists_tables_and_line_breaks_keep_their_lines() {
        let html = b"<ul><li>One</li><li>Two <b>bold</b></li></ul>\
            <table><tr><th>Plan</th><th>Days</th></tr><tr><td>Basic</td><td>30</td></tr></table>\
            <p>Line one<br>Line two</p>";
        assert_eq!(
            extract_text(html),
            "- One\n- Two bold\n\nPlan | Days\nBasic | 30\n\nLine one\nLine two"
        );
    }

    #[test]
    fn preformatted_text_keeps_its_whitespace() {
        let html =
            b"<p>Run:</p><pre>fn main() {\r\n    println!(\"&lt;hi&gt;\");\n}</pre><p>Done.</p>";
        assert_eq!(
            extract_text(html),
            "Run:\n\nfn main() {\n    println!(\"<hi>\");\n}\n\nDone."
        );
    }

    #[test]
    fn a_title_already_leading_the_page_is_not_repeated() {
        let html = b"<title>Guide</title><h1>Guide</h1><p>Body.</p>";
        assert_eq!(extract_text(html), "# Guide\n\nBody.");
    }

    #[test]
    fn a_head_left_open_does_not_hide_the_body() {
        let html = b"<html><head><title>T</title><meta charset=utf-8><body><p>Kept</p>";
        assert_eq!(extract_text(html), "# T\n\nKept");
    }

    #[test]
    fn an_svg_title_is_not_the_page_title() {
        let html = b"<p>Chart:</p><svg><title>tooltip</title><text>axis</text></svg><p>End.</p>";
        assert_eq!(extract_text(html), "Chart:\n\nEnd.");
    }

    #[test]
    fn a_page_with_no_text_reads_as_empty() {
        assert_eq!(
            extract_text(b"<html><head><script>x()</script></head><body> </body></html>"),
            ""
        );
        assert_eq!(extract_text(b""), "");
    }

    #[test]
    fn a_byte_order_mark_and_invalid_utf8_do_not_break_reading() {
        let mut html = b"\xef\xbb\xbf<p>caf\xe9</p>".to_vec();
        html.extend_from_slice(b"<p>ok</p>");
        assert_eq!(extract_text(&html), "caf\u{fffd}\n\nok");
    }
}
