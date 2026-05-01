from sqlalchemy import Column, Integer, String, Float, ForeignKey
from sqlalchemy.orm import relationship
from interfacebench.generators.db_schema import Base

class SimulationRecord(Base):
    __tablename__ = "simulations"
    
    id = Column(Integer, primary_key=True)
    interface_id = Column(Integer, ForeignKey("interfaces.id"), nullable=True)
    slab_id = Column(Integer, ForeignKey("slabs.id"), nullable=True)
    bulk_name = Column(String, nullable=True)     # e.g., 'BaO', 'CuO'
    
    # Engine info
    engine = Column(String)       # 'vasp' or 'ase'
    sim_type = Column(String)     # 'relax' or 'md'
    model_tag = Column(String)    # e.g., 'mace', 'pbe'
    
    # State tracking
    status = Column(String, default="pending") # 'pending', 'queue', 'running', 'complete', 'aborted'
    job_id = Column(String)       # Slurm ID
    calc_folder = Column(String)  # Path to execution directory
    compute_time = Column(Float)  # Seconds

    # Link back to the interface
    interface = relationship("InterfaceRecord", back_populates="simulations")
    slab = relationship("SlabRecord", backref="simulations")