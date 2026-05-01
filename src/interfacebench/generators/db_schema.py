import json
from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey, JSON
from sqlalchemy.orm import declarative_base, relationship
from monty.json import MontyEncoder, MontyDecoder

Base = declarative_base()

class SlabRecord(Base):
    __tablename__ = 'slabs'
    
    id = Column(Integer, primary_key=True)
    base_name = Column(String, nullable=False)
    miller = Column(String, nullable=False)
    termination = Column(String, nullable=False)
    natoms = Column(Integer, nullable=False)
    structure_dict = Column(JSON, nullable=False)  # Stores pymatgen .as_dict()

class InterfaceRecord(Base):
    __tablename__ = 'interfaces'
    
    id = Column(Integer, primary_key=True)
    film_id = Column(Integer, ForeignKey('slabs.id'))
    subs_id = Column(Integer, ForeignKey('slabs.id'))
    natoms = Column(Integer, nullable=False)
    structure_dict = Column(JSON, nullable=False)
    metadata_dict = Column(JSON, nullable=True)
    
    # Establish relational mapping
    film = relationship("SlabRecord", foreign_keys=[film_id])
    subs = relationship("SlabRecord", foreign_keys=[subs_id])

    simulations = relationship("SimulationRecord", back_populates="interface")

def init_db(db_path="structures.db"):
    # Tell SQLAlchemy to use Pymatgen's native encoders
    engine = create_engine(
        f"sqlite:///{db_path}",
        json_serializer=lambda obj: json.dumps(obj, cls=MontyEncoder),
        json_deserializer=lambda obj: json.loads(obj, cls=MontyDecoder)
    )
    Base.metadata.create_all(engine)
    return engine