from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"


class OCRRequest(BaseModel):
    image_base64: str = Field(..., description="Base64-encoded image bytes")


class OCRResponse(BaseModel):
    ocr_text: str


class GradeResult(BaseModel):
    score: float
    max_score: float
    feedback: str
    strengths: list[str] = Field(default_factory=list)
    improvements: list[str] = Field(default_factory=list)
    confidence: float = Field(..., ge=0, le=1)
    rubric_breakdown: dict[str, float] = Field(default_factory=dict)


class GradeTextRequest(BaseModel):
    question: str
    rubric: str
    student_answer: str
    max_score: float = Field(default=10, gt=0)


class GradeTextResponse(BaseModel):
    grade: GradeResult
    raw_model_output: str


class GradeImageRequest(BaseModel):
    image_base64: str
    question: str
    rubric: str
    max_score: float = Field(default=10, gt=0)


class GradeImageResponse(BaseModel):
    ocr_text: str
    grade: GradeResult
    raw_model_output: str


class BulkGradePDFsRequest(BaseModel):
    image_data: str | None = None
    pdf_data: str | None = None
    pdf_files: list[str] = Field(default_factory=list)
    pdf_text: str | None = None
    student_context: str = "General student assessment"
    analysis_type: str = "educational_analysis"
    task_type: str = "grade_assignment"
    assignment_title: str = "Assignment"
    max_points: float = Field(default=100, gt=0)
    grading_rubric: str = "Standard academic rubric"
    student_names: list[str] = Field(default_factory=list)
    feedback_type: str = "both"


class BulkStudentResult(BaseModel):
    student_name: str
    student_index: int
    pdf_pages: int | None = None
    score: float | None = None
    max_points: float | None = None
    letter_grade: str | None = None
    percentage: float | None = None
    feedback: str | None = None
    detailed_feedback: str | None = None
    raw_feedback: str | None = None
    parse_method: str | None = None
    analysis_type: str | None = None
    ocr_text: str | None = None
    warning: str | None = None
    error: str | None = None
    error_detail: str | None = None
    success: bool
    processed_at: str


class BulkGradeSummary(BaseModel):
    total_students: int
    successfully_graded: int
    failed: int
    average_score: float
    max_points: float


class BulkGradePDFsResponse(BaseModel):
    assignment_title: str
    max_points: float
    grading_rubric: str
    total_pdfs: int
    successful: int
    failed: int
    student_results: list[BulkStudentResult]
    batch_summary: BulkGradeSummary
    bulk_processing: bool = True
    analysis_type: str = "ocr_plus_llm"
    success: bool = True


class ErrorResponse(BaseModel):
    detail: str
    meta: dict[str, Any] | None = None
