import logging
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.structure import Structure
from sqlalchemy.orm import sessionmaker
from .db_schema import init_db, SlabRecord, InterfaceRecord

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, db_path="structures.db"):
        self.engine = init_db(db_path)
        self.SessionLocal = sessionmaker(bind=self.engine)
        self.matcher = StructureMatcher()

    def deduplicate_and_save_slabs(self, generated_slabs: list[dict]):
        """
        Deduplicates a list of slab dictionaries in memory, grouped by natoms,
        and saves unique records to the database.
        """
        unique_slabs = []
        
        # Group by natoms to drastically reduce StructureMatcher O(N^2) load
        grouped_by_atoms = {}
        for slab in generated_slabs:
            grouped_by_atoms.setdefault(slab['natoms'], []).append(slab)
            
        for natoms, group in grouped_by_atoms.items():
            for new_slab_dict in group:
                new_struct = Structure.from_dict(new_slab_dict['structure_dict'])
                is_duplicate = False
                
                # Only compare against accepted unique slabs of the exact same size
                for accepted_slab_dict in [s for s in unique_slabs if s['natoms'] == natoms]:
                    accepted_struct = Structure.from_dict(accepted_slab_dict['structure_dict'])
                    
                    if self.matcher.fit(new_struct, accepted_struct):
                        is_duplicate = True
                        break
                        
                if not is_duplicate:
                    unique_slabs.append(new_slab_dict)

        # Save to database
        with self.SessionLocal() as session:
            for slab_data in unique_slabs:
                record = SlabRecord(
                    base_name=slab_data['base_name'],
                    miller=slab_data['miller'],
                    termination=slab_data['termination'],
                    natoms=slab_data['natoms'],
                    structure_dict=slab_data['structure_dict']
                )
                session.add(record)
            session.commit()
            logger.info(f"Saved {len(unique_slabs)} unique slabs out of {len(generated_slabs)} generated.")

    def deduplicate_and_save_interfaces(self, generated_interfaces: list[dict]):
        """
        Deduplicates a list of interface dictionaries, checks against the database,
        and saves unique records, linking them to existing slab records.
        """
        if not generated_interfaces:
            logger.info("No interfaces were generated, nothing to save.")
            return

        unique_interfaces_to_add = []

        # Group by natoms to reduce StructureMatcher O(N^2) load
        grouped_by_atoms = {}
        for interface in generated_interfaces:
            grouped_by_atoms.setdefault(interface['natoms'], []).append(interface)

        with self.SessionLocal() as session:
            # In-memory deduplication first
            in_memory_unique_interfaces = []
            for natoms, group in grouped_by_atoms.items():
                for new_int_dict in group:
                    new_struct = Structure.from_dict(new_int_dict['structure_dict'])
                    is_duplicate = False
                    for accepted_int_dict in [i for i in in_memory_unique_interfaces if i['natoms'] == natoms]:
                        accepted_struct = Structure.from_dict(accepted_int_dict['structure_dict'])
                        if self.matcher.fit(new_struct, accepted_struct):
                            is_duplicate = True
                            break
                    if not is_duplicate:
                        in_memory_unique_interfaces.append(new_int_dict)

            # Deduplicate against the database and prepare records
            for int_data in in_memory_unique_interfaces:
                new_struct = Structure.from_dict(int_data['structure_dict'])

                # Check against DB for duplicates
                existing_records = session.query(InterfaceRecord).filter(InterfaceRecord.natoms == int_data['natoms']).all()
                if any(self.matcher.fit(new_struct, 
                                        rec.structure_dict if isinstance(rec.structure_dict, Structure) 
                                        else Structure.from_dict(rec.structure_dict)) 
                       for rec in existing_records):
                    continue

                # Find parent slabs by base_name, miller, and termination
                film_record = session.query(SlabRecord).filter_by(
                    base_name=int_data['film_name'], miller=int_data['film_miller'],
                    termination=str(int_data['film_termination'])
                ).first()

                subs_record = session.query(SlabRecord).filter_by(
                    base_name=int_data['subs_name'], miller=int_data['subs_miller'],
                    termination=str(int_data['subs_termination'])
                ).first()

                film_id = film_record.id if film_record else None
                subs_id = subs_record.id if subs_record else None

                if film_id is None or subs_id is None:
                    logger.info(f"Missing parent slabs for interface between "
                                   f"{int_data['film_name']}({int_data['film_miller']}) and "
                                   f"{int_data['subs_name']}({int_data['subs_miller']}). "
                                   f"Saving interface without slab mapping.")

                new_interface = InterfaceRecord(
                    film_id=film_id,
                    subs_id=subs_id,
                    natoms=int_data['natoms'],
                    structure_dict=int_data['structure_dict'],
                    metadata_dict=int_data.get('metadata_dict', {})
                )
                session.add(new_interface)
                unique_interfaces_to_add.append(new_interface)

            session.commit()
            logger.info(f"Saved {len(unique_interfaces_to_add)} unique interfaces to the database.")