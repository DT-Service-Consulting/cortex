class Punctuality:

    def get_distinct_platforms_sql_query(self):
        return r'''
            SELECT DISTINCT PTCAR_ID, TRACK_SECTION_ID, TRACK_SECTION_SH_NM_NL
            FROM dbo.public_punctuality_for_simulation
            WHERE PTCAR_ID IN ('215', '216', '217', '220', '221');
        '''
    
    def get_train_no_from_relations_ilike(self,relation_name):
        return f"""
            SELECT DISTINCT p.TRAIN_NO, p.PTCAR_NO, p.REAL_DATETIME_ARR, p.REAL_DATETIME_DEP
            FROM dbo.infra_private_punctuality p
            WHERE p.TRAIN_NO IN (
                SELECT DISTINCT TRAIN_NO
                FROM dbo.public_ns_punctuality
                WHERE RELATION LIKE '{relation_name}'
            )
        """