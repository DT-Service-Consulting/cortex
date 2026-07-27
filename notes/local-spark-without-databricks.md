# Running Spark Locally (Without the Databricks Cluster)

## Problem

With the Databricks VS Code extension, the Jupyter kernel injects environment variables (via `.databricks/.databricks.env`):

- `DATABRICKS_HOST`
- `DATABRICKS_SERVERLESS_COMPUTE_ID=auto`
- and related `DATABRICKS_*` / `SPARK_*` keys

As a result, `SparkSession.builder.getOrCreate()` connects to the **remote cluster** (Spark Connect). A local path like `/Users/melihtaki/...` is then resolved on Databricks (DBFS) → `DBFS_DISABLED` error.

## Solution

Two steps:

### 1. Disconnect from Databricks

Clear the injected variables **before** creating the session:

```python
for key in list(os.environ):
    if key.startswith("DATABRICKS_") or key.startswith("SPARK_"):
        os.environ.pop(key, None)
```

Without this, Spark keeps targeting the cloud workspace.

### 2. Start a local Spark session

With `databricks-connect` installed, classic JVM mode is blocked. Use Spark Connect in local mode instead:

```python
spark = (
    SparkSession.builder
    .remote("local[*]")
    .appName("ETL_private_punctuality")
    .getOrCreate()
)
```

`local[*]` runs a mini Spark cluster on your machine (all cores). Local CSV paths become readable again because the “cluster” is your Mac.

Set `JAVA_HOME` to Temurin 21 (required to start Spark).

## Reading data

No lake upload / `abfss://` needed:

```python
data_link = os.path.abspath("../data/TRAIN_PTCAR_202501.csv")
df = spark.read.csv(data_link, header=True, sep=";")
df.show()
```

The separator is `;` (not `,`).

## Summary

| Mode | Session | Where the file lives |
|------|---------|----------------------|
| Databricks Connect | `getOrCreate()` + `DATABRICKS_*` env | Volume / ADLS (`abfss://`) |
| Local | clear env + `.remote("local[*]")` | Mac path (`/Users/...`) |

Restart the **kernel** after switching, otherwise the previous Databricks session stays active.
