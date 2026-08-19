
-- Pour sauvegarder ce résultat en local au format CSV,
-- ajoutez la commande \copy (dans psql) avant la requête :

\copy (
    SELECT 
        p.*,      -- abréviation BTS
        i.Abbreviation_BTS_French_complete    -- nom complet en français
    FROM public_ns_punctuality p
    LEFT JOIN infra_operational_points i
        ON p.PTCAR_ID = i.PTCAR_ID
    ORDER BY COALESCE(p.PLANNED_DATETIME_DEP, p.PLANNED_DATETIME_ARR, p.PLANNED_DATETIME_DEP)
) TO '/Users/melihtaki/Local/cortex/resultat.csv' WITH CSV HEADER;

-- Remplacez '/chemin/vers/votre/dossier/resultat.csv' par le chemin souhaité sur votre machine.
-- Cette commande doit être exécutée dans le client psql.
