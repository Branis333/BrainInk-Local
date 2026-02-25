import base64
import importlib
import io
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gradio_client import Client, handle_file
from huggingface_hub import InferenceClient

from schema import (
    BulkGradePDFsRequest,
    BulkGradePDFsResponse,
    BulkGradeSummary,
    BulkStudentResult,
    GradeResult,
)


class GradingService:
    def __init__(self) -> None:
        self.hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        self.ocr_space_id = os.getenv("OCR_SPACE_ID", "Branis333/hand_writing_ocr")
        self.ocr_api_name = (os.getenv("OCR_API_NAME", "") or "").strip() or None
        self.ocr_prompt = os.getenv("OCR_PROMPT", "<image>\nFree OCR.")

        self.grader_model = os.getenv("GRADER_MODEL", "Qwen/Qwen2.5-7B-Instruct")
        self.temperature = float(os.getenv("GRADER_TEMPERATURE", "0.0"))
        self.max_new_tokens = int(os.getenv("GRADER_MAX_NEW_TOKENS", "600"))

        self._ocr_client: Client | None = None
        self._llm_client: InferenceClient | None = None
        self._resolved_ocr_api_name: str | None = None

    def _get_ocr_client(self) -> Client:
        if not self.hf_token:
            raise RuntimeError("HF_TOKEN is missing. Set it in .env or environment variables.")

        if self._ocr_client is None:
            try:
                self._ocr_client = Client(self.ocr_space_id, hf_token=self.hf_token)
            except TypeError:
                self._ocr_client = Client(self.ocr_space_id, token=self.hf_token)
        return self._ocr_client

    def _get_llm_client(self) -> InferenceClient:
        if self._llm_client is None:
            self._llm_client = InferenceClient(token=self.hf_token)
        return self._llm_client

    @staticmethod
    def _extract_api_names_from_obj(data: Any) -> list[str]:
        found: list[str] = []

        def walk(obj: Any) -> None:
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if key == "api_name" and isinstance(value, str):
                        found.append(value)
                    walk(value)
                return
            if isinstance(obj, list):
                for item in obj:
                    walk(item)

        walk(data)
        return found

    @staticmethod
    def _unique_preserve_order(values: list[str]) -> list[str]:
        unique: list[str] = []
        seen: set[str] = set()
        for value in values:
            if not value or value in seen:
                continue
            unique.append(value)
            seen.add(value)
        return unique

    def _discover_api_names(self, client: Client) -> list[str]:
        candidates: list[str] = []
        try:
            info = client.view_api(all_endpoints=True)
        except TypeError:
            info = client.view_api()
        except Exception:
            return []

        if isinstance(info, (dict, list)):
            candidates.extend(self._extract_api_names_from_obj(info))
            text = json.dumps(info)
        else:
            text = str(info)

        for match in re.findall(r"api_name\s*[:=]\s*[`\"']([^`\"']+)[`\"']", text):
            candidates.append(match)

        for match in re.findall(r"(/[^\s,`\"'\]\)]+)", text):
            if match.startswith("/"):
                candidates.append(match)

        return self._unique_preserve_order(candidates)

    def _ocr_api_candidates(self, client: Client) -> list[str | None]:
        choices: list[str | None] = []
        if self._resolved_ocr_api_name:
            choices.append(self._resolved_ocr_api_name)
        if self.ocr_api_name:
            choices.append(self.ocr_api_name)

        choices.extend(["/predict", "/run/predict", "/ocr", "/transcribe"])
        choices.extend(self._discover_api_names(client))
        choices.append(None)

        normalized: list[str | None] = []
        seen: set[str] = set()
        for choice in choices:
            if choice is None:
                if None not in normalized:
                    normalized.append(None)
                continue
            if choice not in seen:
                normalized.append(choice)
                seen.add(choice)

        return normalized

    @staticmethod
    def _normalize_ocr_result(result: Any) -> str:
        if isinstance(result, (list, tuple)) and result:
            return str(result[0]).strip()
        return str(result).strip()

    def _call_ocr_predict(self, client: Client, image_path: Path) -> str:
        call_errors: list[str] = []
        file_arg = handle_file(str(image_path))

        for api_name in self._ocr_api_candidates(client):
            kwargs = {"image": file_arg, "prompt": self.ocr_prompt}
            if api_name is not None:
                kwargs["api_name"] = api_name

            try:
                result = client.predict(**kwargs)
                if api_name:
                    self._resolved_ocr_api_name = api_name
                return self._normalize_ocr_result(result)
            except Exception as exc:
                call_errors.append(f"keyword(image,prompt) api={api_name}: {exc}")

            try:
                if api_name is None:
                    result = client.predict(file_arg, self.ocr_prompt)
                else:
                    result = client.predict(file_arg, self.ocr_prompt, api_name=api_name)
                if api_name:
                    self._resolved_ocr_api_name = api_name
                return self._normalize_ocr_result(result)
            except Exception as exc:
                call_errors.append(f"positional(image,prompt) api={api_name}: {exc}")

            try:
                if api_name is None:
                    result = client.predict(file_arg)
                else:
                    result = client.predict(file_arg, api_name=api_name)
                if api_name:
                    self._resolved_ocr_api_name = api_name
                return self._normalize_ocr_result(result)
            except Exception as exc:
                call_errors.append(f"positional(image) api={api_name}: {exc}")

        summary = " | ".join(call_errors[-6:])
        raise RuntimeError(f"No compatible OCR endpoint call succeeded. Recent errors: {summary}")

    @staticmethod
    def _decode_image_to_temp_file(image_base64: str) -> Path:
        raw = base64.b64decode(image_base64, validate=True)
        return GradingService._write_image_bytes_to_temp_file(raw)

    @staticmethod
    def _write_image_bytes_to_temp_file(image_bytes: bytes) -> Path:
        tmpdir = tempfile.mkdtemp(prefix="grading_ocr_")
        path = Path(tmpdir) / "input.png"
        path.write_bytes(image_bytes)
        return path

    @staticmethod
    def _extract_first_json(text: str) -> dict:
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("Could not find JSON object in model output")

        candidate = text[start : end + 1]
        return json.loads(candidate)

    @staticmethod
    def _extract_chat_text(response: Any) -> str:
        if response is None:
            return ""

        if isinstance(response, str):
            return response

        if isinstance(response, dict):
            choices = response.get("choices") or []
            if choices:
                message = choices[0].get("message") or {}
                content = message.get("content")
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    chunks: list[str] = []
                    for item in content:
                        if isinstance(item, dict) and isinstance(item.get("text"), str):
                            chunks.append(item["text"])
                    if chunks:
                        return "\n".join(chunks)
            if isinstance(response.get("generated_text"), str):
                return response["generated_text"]
            return json.dumps(response)

        if hasattr(response, "choices"):
            choices = getattr(response, "choices", None)
            if choices:
                first = choices[0]
                message = getattr(first, "message", None)
                if message is not None:
                    content = getattr(message, "content", None)
                    if isinstance(content, str):
                        return content
                    if isinstance(content, list):
                        chunks: list[str] = []
                        for item in content:
                            text = getattr(item, "text", None)
                            if isinstance(text, str):
                                chunks.append(text)
                        if chunks:
                            return "\n".join(chunks)

        return str(response)

    def _generate_grade_output(self, prompt: str) -> str:
        client = self._get_llm_client()
        failures: list[str] = []

        try:
            return client.text_generation(
                prompt=prompt,
                model=self.grader_model,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                return_full_text=False,
            )
        except Exception as exc:
            failures.append(f"text_generation: {exc}")

        messages = [
            {
                "role": "system",
                "content": "You are a strict academic grader. Return only valid JSON.",
            },
            {"role": "user", "content": prompt},
        ]

        try:
            response = client.chat.completions.create(
                model=self.grader_model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_new_tokens,
            )
            text = self._extract_chat_text(response)
            if text.strip():
                return text
            failures.append("chat.completions.create: empty response")
        except Exception as exc:
            failures.append(f"chat.completions.create: {exc}")

        try:
            response = client.chat_completion(
                model=self.grader_model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_new_tokens,
            )
            text = self._extract_chat_text(response)
            if text.strip():
                return text
            failures.append("chat_completion: empty response")
        except Exception as exc:
            failures.append(f"chat_completion: {exc}")

        raise RuntimeError("No grading inference method succeeded. " + " | ".join(failures[-3:]))

    @staticmethod
    def _coerce_grade(payload: dict, max_score: float) -> GradeResult:
        score = float(payload.get("score", 0.0))
        score = max(0.0, min(score, max_score))
        confidence = float(payload.get("confidence", 0.5))
        confidence = max(0.0, min(confidence, 1.0))

        strengths = payload.get("strengths", [])
        if not isinstance(strengths, list):
            strengths = [str(strengths)]

        improvements = payload.get("improvements", [])
        if not isinstance(improvements, list):
            improvements = [str(improvements)]

        rubric_breakdown = payload.get("rubric_breakdown", {})
        if not isinstance(rubric_breakdown, dict):
            rubric_breakdown = {}

        return GradeResult(
            score=score,
            max_score=max_score,
            feedback=str(payload.get("feedback", "")),
            strengths=[str(item) for item in strengths],
            improvements=[str(item) for item in improvements],
            confidence=confidence,
            rubric_breakdown={str(k): float(v) for k, v in rubric_breakdown.items() if str(v).strip()},
        )

    def run_ocr(self, image_base64: str) -> str:
        image_path = self._decode_image_to_temp_file(image_base64)
        return self.run_ocr_from_path(image_path)

    def run_ocr_bytes(self, image_bytes: bytes) -> str:
        image_path = self._write_image_bytes_to_temp_file(image_bytes)
        return self.run_ocr_from_path(image_path)

    def run_ocr_from_path(self, image_path: Path) -> str:
        client = self._get_ocr_client()
        return self._call_ocr_predict(client, image_path)

    def grade_text(self, question: str, rubric: str, student_answer: str, max_score: float) -> tuple[GradeResult, str]:
        prompt = f"""You are a strict academic grader.
Return ONLY valid JSON with this exact schema:
{{
  \"score\": number,
  \"feedback\": string,
  \"strengths\": string[],
  \"improvements\": string[],
  \"confidence\": number,
  \"rubric_breakdown\": {{\"criterion\": number}}
}}

Rules:
- Score range is 0 to {max_score}.
- Be fair and concise.
- Do not include markdown or extra text.

Question:
{question}

Rubric:
{rubric}

Student Answer:
{student_answer}
"""

        raw_output = self._generate_grade_output(prompt)

        parsed = self._extract_first_json(raw_output)
        grade = self._coerce_grade(parsed, max_score=max_score)
        return grade, raw_output

    def grade_image(self, image_base64: str, question: str, rubric: str, max_score: float) -> tuple[str, GradeResult, str]:
        ocr_text = self.run_ocr(image_base64)
        grade, raw_output = self.grade_text(
            question=question,
            rubric=rubric,
            student_answer=ocr_text,
            max_score=max_score,
        )
        return ocr_text, grade, raw_output

    def grade_image_bytes(self, image_bytes: bytes, question: str, rubric: str, max_score: float) -> tuple[str, GradeResult, str]:
        ocr_text = self.run_ocr_bytes(image_bytes)
        grade, raw_output = self.grade_text(
            question=question,
            rubric=rubric,
            student_answer=ocr_text,
            max_score=max_score,
        )
        return ocr_text, grade, raw_output

    @staticmethod
    def _to_letter_grade(score: float, max_points: float) -> str:
        if max_points <= 0:
            return "F"
        percentage = (score / max_points) * 100
        if percentage >= 97:
            return "A+"
        if percentage >= 93:
            return "A"
        if percentage >= 90:
            return "A-"
        if percentage >= 87:
            return "B+"
        if percentage >= 83:
            return "B"
        if percentage >= 80:
            return "B-"
        if percentage >= 77:
            return "C+"
        if percentage >= 73:
            return "C"
        if percentage >= 70:
            return "C-"
        if percentage >= 67:
            return "D+"
        if percentage >= 63:
            return "D"
        if percentage >= 60:
            return "D-"
        return "F"

    @staticmethod
    def _decode_base64_bytes(payload: str) -> bytes:
        return base64.b64decode(payload, validate=True)

    def _pdf_bytes_to_page_images(self, pdf_bytes: bytes) -> list[bytes]:
        images: list[bytes] = []
        errors: list[str] = []

        try:
            pdfium = importlib.import_module("pypdfium2")

            document = pdfium.PdfDocument(pdf_bytes)
            for page_index in range(len(document)):
                page = document[page_index]
                bitmap = page.render(scale=2)
                pil_image = bitmap.to_pil()
                buffer = io.BytesIO()
                pil_image.save(buffer, format="PNG")
                images.append(buffer.getvalue())
                page.close()
            document.close()
            if images:
                return images
            errors.append("pypdfium2 produced zero pages")
        except Exception as exc:
            errors.append(f"pypdfium2: {exc}")

        try:
            fitz = importlib.import_module("fitz")

            document = fitz.open(stream=pdf_bytes, filetype="pdf")
            for page in document:
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                images.append(pix.tobytes("png"))
            document.close()
            if images:
                return images
            errors.append("pymupdf produced zero pages")
        except Exception as exc:
            errors.append(f"pymupdf: {exc}")

        raise RuntimeError(
            "Failed to convert PDF pages to images. Install either pypdfium2 or pymupdf. "
            + " | ".join(errors)
        )

    def _pdf_base64_to_page_images(self, pdf_base64: str) -> list[bytes]:
        pdf_bytes = self._decode_base64_bytes(pdf_base64)
        return self._pdf_bytes_to_page_images(pdf_bytes)

    @staticmethod
    def _utc_now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    def bulk_grade_pdfs(self, payload: BulkGradePDFsRequest) -> BulkGradePDFsResponse:
        pdf_files = list(payload.pdf_files)
        if not pdf_files and payload.pdf_data:
            pdf_files = [payload.pdf_data]

        if not pdf_files and payload.pdf_text:
            grade, raw_output = self.grade_text(
                question=f"Grade this submission for assignment: {payload.assignment_title}",
                rubric=payload.grading_rubric,
                student_answer=payload.pdf_text,
                max_score=payload.max_points,
            )
            score = round(float(grade.score), 2)
            result = BulkStudentResult(
                student_name=(payload.student_names[0] if payload.student_names else "Student 1"),
                student_index=0,
                pdf_pages=None,
                score=score,
                max_points=payload.max_points,
                letter_grade=self._to_letter_grade(score, payload.max_points),
                percentage=round((score / payload.max_points) * 100, 2) if payload.max_points > 0 else 0.0,
                feedback=grade.feedback,
                detailed_feedback=grade.feedback,
                raw_feedback=raw_output,
                parse_method="json_schema",
                analysis_type="text_only",
                ocr_text=payload.pdf_text,
                success=True,
                processed_at=self._utc_now_iso(),
            )
            return BulkGradePDFsResponse(
                assignment_title=payload.assignment_title,
                max_points=payload.max_points,
                grading_rubric=payload.grading_rubric,
                total_pdfs=1,
                successful=1,
                failed=0,
                student_results=[result],
                batch_summary=BulkGradeSummary(
                    total_students=1,
                    successfully_graded=1,
                    failed=0,
                    average_score=score,
                    max_points=payload.max_points,
                ),
                analysis_type="ocr_plus_llm",
                success=True,
            )

        if not pdf_files:
            raise RuntimeError("Either pdf_files or pdf_data or pdf_text is required")

        results: list[BulkStudentResult] = []
        successful_scores: list[float] = []

        for idx, pdf_base64 in enumerate(pdf_files):
            student_name = (
                payload.student_names[idx].strip()
                if idx < len(payload.student_names) and payload.student_names[idx].strip()
                else f"Student {idx + 1}"
            )

            try:
                page_images = self._pdf_base64_to_page_images(pdf_base64)
                page_texts: list[str] = []

                for page_idx, image_bytes in enumerate(page_images):
                    page_text = self.run_ocr_bytes(image_bytes)
                    clean_text = page_text.strip()
                    if clean_text:
                        page_texts.append(f"[Page {page_idx + 1}]\n{clean_text}")

                merged_text = "\n\n".join(page_texts).strip()
                if not merged_text:
                    raise RuntimeError("OCR returned empty text for all pages")

                question = (
                    f"Grade the student's submission for assignment '{payload.assignment_title}'. "
                    f"Use the rubric strictly and justify deductions with concise evidence from the submission."
                )
                grade, raw_output = self.grade_text(
                    question=question,
                    rubric=payload.grading_rubric,
                    student_answer=merged_text,
                    max_score=payload.max_points,
                )

                score = round(float(grade.score), 2)
                successful_scores.append(score)

                results.append(
                    BulkStudentResult(
                        student_name=student_name,
                        student_index=idx,
                        pdf_pages=len(page_images),
                        score=score,
                        max_points=payload.max_points,
                        letter_grade=self._to_letter_grade(score, payload.max_points),
                        percentage=round((score / payload.max_points) * 100, 2) if payload.max_points > 0 else 0.0,
                        feedback=grade.feedback,
                        detailed_feedback=grade.feedback,
                        raw_feedback=raw_output,
                        parse_method="json_schema",
                        analysis_type="ocr_plus_llm",
                        ocr_text=merged_text,
                        success=True,
                        processed_at=self._utc_now_iso(),
                    )
                )
            except Exception as exc:
                results.append(
                    BulkStudentResult(
                        student_name=student_name,
                        student_index=idx,
                        analysis_type="ocr_plus_llm",
                        error="grading_failed",
                        error_detail=str(exc),
                        success=False,
                        processed_at=self._utc_now_iso(),
                    )
                )

        successful = len([row for row in results if row.success])
        failed = len(results) - successful
        average_score = round(sum(successful_scores) / successful, 2) if successful else 0.0

        return BulkGradePDFsResponse(
            assignment_title=payload.assignment_title,
            max_points=payload.max_points,
            grading_rubric=payload.grading_rubric,
            total_pdfs=len(pdf_files),
            successful=successful,
            failed=failed,
            student_results=results,
            batch_summary=BulkGradeSummary(
                total_students=len(results),
                successfully_graded=successful,
                failed=failed,
                average_score=average_score,
                max_points=payload.max_points,
            ),
            analysis_type="ocr_plus_llm",
            success=True,
        )
