from pydantic import BaseModel, Field, field_validator
from typing import Optional


class BankDetails(BaseModel):
    bank_name: str = Field(..., description="Название банка")
    bik: str = Field("", min_length=9)
    account_number: str = Field(..., pattern=r'^\d{20}$', description="Расчетный счет (20 цифр)")
    correspondent_account: str = Field(..., pattern=r'^\d{20}$', description="Корреспондентский счет (20 цифр)")



class CompanyRequisites(BaseModel):
    full_legal_name: str = Field(..., min_length=1)
    inn: str = Field(..., pattern=r'^\d{10,12}$')
    ogrn_or_ogrnip: str = Field(..., pattern=r'^\d{13}')
    legal_address: str = Field(..., min_length=1)
    signatory: str = Field("", description="Подпись руководителя")
    bank_details: BankDetails

    @field_validator('inn')
    @classmethod
    def validate_inn(cls, v):
        if not (len(v) == 10 or len(v) == 12):
            raise ValueError("ИНН должен быть 10 или 12 символов")
        return v



class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str

    @field_validator('role')
    @classmethod
    def validate_role(cls, v):
        if v not in ["user", "assistant", "system"]:
            raise ValueError("Role must be user, assistant, or system")
        return v


class ConversationHistory(BaseModel):
    timestamp_start: str = Field(default_factory=lambda: __import__('datetime').datetime.now().isoformat())
    llm_endpoint_used: Optional[str] = None
    company_details: Optional[CompanyRequisites] = None
    messages: list[ChatMessage] = []

    def to_dict(self) -> dict:
        return self.model_dump(exclude_none=True, by_alias=True)

    @classmethod
    def from_dict(cls, data: dict):
        # Валидация при загрузке истории
        parsed_data = {}

        if "company_details" in data and isinstance(data["company_details"], dict):
            try:
                parsed_data["company_details"] = CompanyRequisites(**data["company_details"])

            except Exception as e:
                # Если данные повреждены, игнорируем реквизиты при загрузке
                pass
        

        if "messages" in data and isinstance(data["messages"], list):
            try:
                parsed_data["messages"] = [ChatMessage(**msg) for msg in data["messages"]]
            except Exception:
                parsed_data["messages"] = []

        # Остальные поля берем как есть или дефолты
        parsed_data.update(data)
        
        return cls(**parsed_data)

