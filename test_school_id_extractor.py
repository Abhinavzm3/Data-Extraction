import unittest

from school_id_extractor import SchoolIDExtractor


class SchoolIDExtractorParsingTests(unittest.TestCase):
    def test_extracts_all_fields_from_labeled_lines(self):
        lines = [
            "Motilal Nehru National Institute of Technology Allahabad",
            "IDENTITY CARD",
            "Name : Abhinav Kumar Maurya",
            "Enrollment No.: 20234181",
            "Programme : Bachelor of Technology",
            "Department : Electronics & Comm. Engg.",
        ]

        result = SchoolIDExtractor.parse_lines(lines)

        self.assertEqual(result["name"], "Abhinav Kumar Maurya")
        self.assertEqual(result["enrollment_no"], "20234181")
        self.assertEqual(result["programme"], "Bachelor of Technology")
        self.assertEqual(result["department"], "Electronics & Comm. Engg")

    def test_extracts_values_when_label_and_value_are_split(self):
        lines = [
            "Name",
            "Abhinav Kumar Maurya",
            "Enrollment No.",
            "20234181",
            "Programme",
            "Bachelor of Technology",
            "Department",
            "Electronics & Comm. Engg.",
        ]

        result = SchoolIDExtractor.parse_lines(lines)

        self.assertEqual(result["name"], "Abhinav Kumar Maurya")
        self.assertEqual(result["enrollment_no"], "20234181")
        self.assertEqual(result["programme"], "Bachelor of Technology")
        self.assertEqual(result["department"], "Electronics & Comm. Engg")


if __name__ == "__main__":
    unittest.main()
