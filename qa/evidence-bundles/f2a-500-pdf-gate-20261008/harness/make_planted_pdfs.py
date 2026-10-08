"""Generate the 50 planted-fact PDFs for the F2a 500-PDF gate.

Each document is a few pages of plain report prose that carries one planted
fact, and a near-duplicate decoy of the next document's fact (same entity,
different attribute and number), so a retriever that ranks by surface overlap
alone can pick the wrong one. The PDFs are written by hand with the standard
Helvetica font so the generator needs nothing beyond Python.

Usage: python3 make_planted_pdfs.py <out dir>
Writes <out dir>/planted/*.pdf and <out dir>/planted-facts.json.
"""
import json
import os
import random
import sys

OUT = sys.argv[1]
os.makedirs(os.path.join(OUT, "planted"), exist_ok=True)
rng = random.Random(20261008)

PLACES = ["Varnholt", "Kessring", "Oduya", "Marlowe Fen", "Tarquin Ridge", "Halvesund", "Brightwater",
          "Corvane", "Ellismere", "Joroni", "Quillon", "Sabreth", "Ulmstead", "Wyncroft", "Zephara",
          "Aldermoor", "Belcastro", "Drovenik", "Fairhollow", "Gantry Point", "Hesketh", "Ivarsdal",
          "Lowenmark", "Norbury", "Pellacre"]
FACILITIES = [("observatory", "primary mirror", "secondary mirror", "recoated"),
              ("water treatment plant", "north clarifier", "south clarifier", "relined"),
              ("rail depot", "east turntable", "west turntable", "rebuilt")]
SUBJECTS = ["maintenance scheduling", "budget planning", "staff rotation", "procurement review",
            "safety inspection", "energy usage", "visitor access", "records retention"]
FILLER = [
    "The {s} committee met quarterly and recorded its decisions in the shared register.",
    "Contractors were asked to submit revised estimates before the end of each review period.",
    "Several recommendations from the previous audit were carried forward without change.",
    "Staff noted that the {s} process depended on paper forms that were often misfiled.",
    "A pilot programme replaced two manual steps with a single checklist reviewed by a supervisor.",
    "The board asked for a summary of outstanding actions to be circulated a week before each meeting.",
    "Weather delays affected the schedule in the winter months, as in most previous years.",
    "Feedback from site managers was collected through short structured interviews.",
    "The finance office reconciled invoices against purchase orders at the end of every month.",
    "An external reviewer found the documentation adequate but recommended clearer version control.",
    "Training sessions were offered in the mornings to avoid conflicts with operational duties.",
    "The report on {s} was approved with minor amendments to its appendix tables.",
]

facts = []
for i in range(50):
    place = PLACES[i % len(PLACES)]
    kind, part, decoy_part, verb = FACILITIES[i % len(FACILITIES)]
    amount = rng.randrange(120, 980) * 1000 + rng.randrange(1, 9) * 100
    decoy_amount = rng.randrange(120, 980) * 1000 + rng.randrange(1, 9) * 100
    year, decoy_year = rng.randrange(2009, 2024), rng.randrange(2009, 2024)
    name = f"{place} {kind}"
    facts.append({
        "id": i,
        "file": f"planted-{i:02d}.pdf",
        "fact": f"The {part} of the {name} was {verb} in {year} at a cost of {amount:,} euros.",
        "decoy": f"The {decoy_part} of the {name} was {verb} in {decoy_year} at a cost of {decoy_amount:,} euros.",
        "question": f"How much did it cost when the {part} of the {name} was {verb}?",
        "answer": f"{amount:,}",
        "decoy_answer": f"{decoy_amount:,}",
    })


def paragraphs(doc_rng, count):
    out = []
    for _ in range(count):
        subject = doc_rng.choice(SUBJECTS)
        sentences = [doc_rng.choice(FILLER).format(s=subject) for _ in range(doc_rng.randrange(4, 7))]
        out.append(" ".join(sentences))
    return out


def wrap(text, width=92):
    lines, line = [], ""
    for word in text.split():
        if line and len(line) + 1 + len(word) > width:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}" if line else word
    if line:
        lines.append(line)
    return lines


def pdf_escape(text):
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def write_pdf(path, title, body_paragraphs):
    lines = [("title", title), ("blank", "")]
    for para in body_paragraphs:
        lines += [("text", l) for l in wrap(para)] + [("blank", "")]
    per_page = 48
    pages = [lines[i:i + per_page] for i in range(0, len(lines), per_page)]
    objects = []  # index 0 -> obj 1

    def add(body):
        objects.append(body)
        return len(objects)

    catalog = add(None)
    pages_obj = add(None)
    font = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>")
    bold = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>")
    page_ids = []
    for page in pages:
        ops = ["BT", "14 TL", "72 760 Td"]
        for kind, text in page:
            if kind == "title":
                ops += ["/F2 14 Tf", f"({pdf_escape(text)}) Tj", "T*"]
            elif kind == "text":
                ops += ["/F1 10 Tf", f"({pdf_escape(text)}) Tj", "T*"]
            else:
                ops += ["T*"]
        ops.append("ET")
        stream = "\n".join(ops).encode("cp1252")
        content = add(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
        page_ids.append(add(
            b"<< /Type /Page /Parent %d 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 %d 0 R /F2 %d 0 R >> >> /Contents %d 0 R >>"
            % (pages_obj, font, bold, content)))
    objects[catalog - 1] = b"<< /Type /Catalog /Pages %d 0 R >>" % pages_obj
    kids = b" ".join(b"%d 0 R" % p for p in page_ids)
    objects[pages_obj - 1] = b"<< /Type /Pages /Kids [%s] /Count %d >>" % (kids, len(page_ids))

    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for offset in offsets:
        out += b"%010d 00000 n \n" % offset
    out += b"trailer\n<< /Size %d /Root %d 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, catalog, xref)
    open(path, "wb").write(bytes(out))


for i, fact in enumerate(facts):
    doc_rng = random.Random(1000 + i)
    body = paragraphs(doc_rng, 14)
    decoy_of = facts[(i + 1) % len(facts)]
    body.insert(doc_rng.randrange(2, 6), fact["fact"] + " " + " ".join(doc_rng.choice(FILLER).format(s="capital works") for _ in range(2)))
    body.insert(doc_rng.randrange(8, 12), decoy_of["decoy"] + " " + " ".join(doc_rng.choice(FILLER).format(s="capital works") for _ in range(2)))
    write_pdf(os.path.join(OUT, "planted", fact["file"]),
              f"Annual facilities report {2000 + i}: {PLACES[i % len(PLACES)]}", body)

for i, fact in enumerate(facts):
    # Document i - 1 carries the decoy of fact i.
    fact["decoy_file"] = facts[(i - 1) % len(facts)]["file"]

json.dump(facts, open(os.path.join(OUT, "planted-facts.json"), "w"), indent=1)
print(f"wrote {len(facts)} planted PDFs")
