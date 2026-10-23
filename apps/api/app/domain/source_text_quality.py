"""Local byte decoding and body quality signals; never repairs or invents prose."""

import codecs
import re
import unicodedata
from dataclasses import asdict, dataclass

_MOJIBAKE = re.compile(r"Ã.|Â.|â[€\x80-\xbf]|ðŸ|ï¿½|锟斤拷|銆|锛|馃")
_CHARSETS = {"utf-8", "utf-8-sig", "gbk", "gb18030", "cp1252", "iso8859-1"}


@dataclass(frozen=True)
class TextQuality:
    accepted: bool
    replacement_ratio: float
    control_ratio: float
    printable_ratio: float
    substantive_ratio: float
    mojibake_ratio: float
    quality_flag: str

    def metrics(self) -> dict[str, bool | float | str]:
        return asdict(self)


def assess_text_quality(text: str) -> TextQuality:
    size = max(1, len(text))
    replacement = text.count("\ufffd") / size
    control = sum(unicodedata.category(c) == "Cc" and c not in "\n\r\t" for c in text) / size
    printable = sum(c.isprintable() or c in "\n\r\t" for c in text) / size
    substantive = sum(c.isalnum() for c in text) / size
    mojibake = sum(len(m.group()) for m in _MOJIBAKE.finditer(text)) / size
    bad = replacement > 0.01 or control > 0.01 or printable < 0.90 or mojibake > 0.025
    return TextQuality(
        not bad,
        replacement,
        control,
        printable,
        substantive,
        mojibake,
        "corrupt_text" if bad else "text_quality_passed",
    )


def decode_html(
    body: bytes, header_charset: str | None
) -> tuple[str, dict[str, str | bool | float]]:
    """Prefer lossless UTF-8, then bounded declared/legacy candidates. No ignore."""
    meta = re.search(rb"<meta\b[^>]*charset\s*=\s*[\x22\x27\s]*([\w-]+)", body[:16384], re.I)
    declared = meta.group(1).decode("ascii") if meta else None
    names = ["utf-8-sig", declared, header_charset, "gb18030", "cp1252"]
    candidates = []
    seen = set()
    for index, name in enumerate(names):
        if not name:
            continue
        try:
            encoding = codecs.lookup(name).name
        except LookupError:
            continue
        if encoding not in _CHARSETS or encoding in seen:
            continue
        seen.add(encoding)
        try:
            text = body.decode(encoding, errors="strict")
        except UnicodeError:
            continue
        quality = assess_text_quality(text)
        penalty = (
            quality.replacement_ratio * 20 + quality.control_ratio * 10 + quality.mojibake_ratio * 5
        )
        candidates.append((penalty, index, text, encoding))
        # Valid UTF-8 is stronger evidence than a contradictory legacy declaration.
        if index == 0 and quality.accepted:
            break
    if candidates:
        _, index, text, encoding = min(candidates, key=lambda row: (row[0], row[1]))
        flag = "strict_utf8" if index == 0 else "strict_candidate_selection"
    else:
        text, encoding, flag = body.decode("utf-8", errors="replace"), "utf-8", "lossy_fallback"
    return text, {
        "detected_charset": encoding,
        "declared_meta_charset": declared or "unspecified",
        "declared_header_charset": header_charset or "unspecified",
        "decode_quality_flag": flag,
    }
