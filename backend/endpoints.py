from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from schema import (
    BulkGradePDFsRequest,
    BulkGradePDFsResponse,
    GradeImageRequest,
    GradeImageResponse,
    GradeTextRequest,
    GradeTextResponse,
    HealthResponse,
    OCRRequest,
    OCRResponse,
)
from services import GradingService

router = APIRouter(prefix="/api", tags=["grading"])
service = GradingService()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.post("/ocr", response_model=OCRResponse)
def ocr_only(payload: OCRRequest) -> OCRResponse:
    try:
        text = service.run_ocr(payload.image_base64)
        return OCRResponse(ocr_text=text)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"OCR failed: {exc}") from exc


@router.post("/ocr/upload", response_model=OCRResponse)
async def ocr_only_upload(image: UploadFile = File(...)) -> OCRResponse:
    try:
        image_bytes = await image.read()
        text = service.run_ocr_bytes(image_bytes)
        return OCRResponse(ocr_text=text)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"OCR upload failed: {exc}") from exc


@router.post("/grade/text", response_model=GradeTextResponse)
def grade_text(payload: GradeTextRequest) -> GradeTextResponse:
    try:
        grade, raw_output = service.grade_text(
            question=payload.question,
            rubric=payload.rubric,
            student_answer=payload.student_answer,
            max_score=payload.max_score,
        )
        return GradeTextResponse(grade=grade, raw_model_output=raw_output)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Text grading failed: {exc}") from exc


@router.post("/grade/image", response_model=GradeImageResponse)
def grade_image(payload: GradeImageRequest) -> GradeImageResponse:
    try:
        ocr_text, grade, raw_output = service.grade_image(
            image_base64=payload.image_base64,
            question=payload.question,
            rubric=payload.rubric,
            max_score=payload.max_score,
        )
        return GradeImageResponse(ocr_text=ocr_text, grade=grade, raw_model_output=raw_output)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Image grading failed: {exc}") from exc


@router.post("/grade/image/upload", response_model=GradeImageResponse)
async def grade_image_upload(
    image: UploadFile = File(...),
    question: str = Form(...),
    rubric: str = Form(...),
    max_score: float = Form(10),
) -> GradeImageResponse:
    try:
        image_bytes = await image.read()
        ocr_text, grade, raw_output = service.grade_image_bytes(
            image_bytes=image_bytes,
            question=question,
            rubric=rubric,
            max_score=max_score,
        )
        return GradeImageResponse(ocr_text=ocr_text, grade=grade, raw_model_output=raw_output)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Image upload grading failed: {exc}") from exc


@router.post("/kana/bulk-grade-pdfs", response_model=BulkGradePDFsResponse)
def bulk_grade_pdfs(payload: BulkGradePDFsRequest) -> BulkGradePDFsResponse:
    try:
        return service.bulk_grade_pdfs(payload)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Bulk PDF grading failed: {exc}") from exc
