from sqlalchemy import Column, String, Float, Integer
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class GeneExpression(Base):
    __tablename__ = "gene_expression"

    id = Column(Integer, primary_key=True)
    gene = Column(String)
    sample = Column(String)
    group = Column(String)
    expression = Column(Float)

