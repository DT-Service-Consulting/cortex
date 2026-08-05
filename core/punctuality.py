class Punctuality:

    def get_distinct_platforms_sql_query(self):
        return r'''
            SELECT DISTINCT PTCAR_ID, TRACK_SECTION_ID, TRACK_SECTION_SH_NM_NL
            FROM dbo.public_punctuality_for_simulation
            WHERE PTCAR_ID IN ('215', '216', '217', '220', '221');
        '''
    
