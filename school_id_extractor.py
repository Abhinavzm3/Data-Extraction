import re


class SchoolIDExtractor:
    FIELDS = ("name", "enrollment_no", "programme", "department")
    FIELD_LABELS = {
        "name": r"name",
        "enrollment_no": r"enrol+l?ment\s*(?:no\.?|number)?",
        "programme": r"programme|program(?:me)?|course",
        "department": r"department|dept\.?",
    }

    def __init__(self):
        import easyocr

        self.reader = easyocr.Reader(["en"], gpu=False)

    def extract(self, img):
        variants = self._build_variants(img)
        merged_result = {field: "" for field in self.FIELDS}

        for variant in variants:
            lines = self._ocr_lines(variant)
            parsed = self.parse_lines(lines)

            for field in self.FIELDS:
                if not merged_result[field] and parsed[field]:
                    merged_result[field] = parsed[field]

            if all(merged_result.values()):
                break

        return merged_result

    def _build_variants(self, img):
        import cv2

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        upscaled = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(upscaled)
        thresholded = cv2.threshold(
            enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )[1]

        return [img, upscaled, thresholded]

    def _ocr_lines(self, img):
        raw_lines = self.reader.readtext(img, detail=0, paragraph=False)
        lines = []
        seen = set()

        for raw_line in raw_lines:
            normalized = self._normalize_text(raw_line)
            lookup_key = normalized.lower()

            if normalized and lookup_key not in seen:
                seen.add(lookup_key)
                lines.append(normalized)

        return lines

    @classmethod
    def parse_lines(cls, lines):
        cleaned_lines = []
        for line in lines:
            normalized = cls._normalize_text(line)
            if normalized:
                cleaned_lines.append(normalized)

        joined_text = " | ".join(cleaned_lines)
        extracted = {field: "" for field in cls.FIELDS}

        for field in cls.FIELDS:
            extracted[field] = cls._extract_from_joined_text(field, joined_text)

        for index, line in enumerate(cleaned_lines):
            for field in cls.FIELDS:
                if extracted[field]:
                    continue

                inline_value = cls._extract_inline_value(field, line)
                if inline_value:
                    extracted[field] = inline_value
                    continue

                if cls._is_label_only_line(field, line):
                    next_value = cls._find_next_value(cleaned_lines, index + 1)
                    if next_value:
                        extracted[field] = cls._clean_value(field, next_value)

        if not extracted["enrollment_no"]:
            extracted["enrollment_no"] = cls._fallback_enrollment_number(joined_text)

        return extracted

    @classmethod
    def _extract_from_joined_text(cls, field, text):
        field_pattern = cls.FIELD_LABELS[field]
        other_patterns = [
            pattern
            for current_field, pattern in cls.FIELD_LABELS.items()
            if current_field != field
        ]
        next_label_pattern = "|".join(f"(?:{pattern})" for pattern in other_patterns)

        match = re.search(
            rf"(?:^|\|)\s*(?:{field_pattern})\s*[:\-]?\s*(.+?)(?=\s*(?:\||(?:{next_label_pattern})\b)|$)",
            text,
            re.IGNORECASE,
        )
        if not match:
            return ""

        return cls._clean_value(field, match.group(1))

    @classmethod
    def _extract_inline_value(cls, field, line):
        field_pattern = cls.FIELD_LABELS[field]
        match = re.search(
            rf"^\s*(?:{field_pattern})\s*[:\-]?\s*(.+)$",
            line,
            re.IGNORECASE,
        )
        if not match:
            return ""

        return cls._clean_value(field, match.group(1))

    @classmethod
    def _is_label_only_line(cls, field, line):
        field_pattern = cls.FIELD_LABELS[field]
        return bool(
            re.fullmatch(rf"\s*(?:{field_pattern})\s*[:\-]?\s*", line, re.IGNORECASE)
        )

    @classmethod
    def _find_next_value(cls, lines, start_index):
        for line in lines[start_index:]:
            if cls._looks_like_label(line):
                return ""
            if line:
                return line
        return ""

    @classmethod
    def _looks_like_label(cls, line):
        return any(
            cls._is_label_only_line(field, line)
            or re.match(rf"^\s*(?:{pattern})\s*[:\-]?", line, re.IGNORECASE)
            for field, pattern in cls.FIELD_LABELS.items()
        )

    @classmethod
    def _fallback_enrollment_number(cls, text):
        compact_text = text.replace(" ", "")
        match = re.search(
            r"\b(?=[A-Z0-9/-]{6,20}\b)(?=[A-Z0-9/-]*\d)[A-Z0-9/-]+\b",
            compact_text,
            re.IGNORECASE,
        )
        if not match:
            return ""
        return match.group(0)

    @classmethod
    def _clean_value(cls, field, value):
        cleaned = cls._normalize_text(value).strip(" |:-")
        cleaned = re.sub(r"\s+\|\s+", " ", cleaned)

        if field == "enrollment_no":
            cleaned = re.sub(r"[^A-Za-z0-9/-]", "", cleaned)
        else:
            cleaned = re.sub(r"\s+", " ", cleaned)
            cleaned = cleaned.rstrip(".,;:")

        return cleaned

    @staticmethod
    def _normalize_text(text):
        text = str(text)
        text = text.replace("\n", " ")
        text = text.replace("\t", " ")
        text = text.replace("\u2014", "-")
        text = text.replace("\u2013", "-")
        text = text.replace("\u2019", "'")
        text = re.sub(r"\s+", " ", text)
        return text.strip()
