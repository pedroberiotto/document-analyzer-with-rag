class AppError(Exception):
    pass


class DocumentNotFoundError(AppError):
    pass


class InvalidDocumentError(AppError):
    pass


class ExtractionError(AppError):
    pass
