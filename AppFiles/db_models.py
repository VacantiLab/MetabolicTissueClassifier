from sqlalchemy import Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base

# Create a base class for object relational mapping models (ORMs)
Base = declarative_base()

class UploadedData(Base):
    __tablename__ = "uploaded_data"

    id = Column(Integer, primary_key=True, index=True)
    content = Column(String)  # text from uploaded file
