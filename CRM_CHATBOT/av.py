from pyspark.sql import SparkSession
from awsglue.utils import getResolvedOptions
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta
import jaydebeapi
import sys

# =============== Spark ================
spark = SparkSession.builder \
    .config('spark.sql.legacy.parquet.datetimeRebaseModeInRead', 'CORRECTED') \
    .config('spark.sql.files.maxPartitionBytes', 134217728) \
    .enableHiveSupport() \
    .getOrCreate()

# =============== Parâmetros ================
args = getResolvedOptions(sys.argv, ['MES'])

today = (date.today() - timedelta(days=1) + relativedelta(months=-int(args['MES'])))
year_str = today.strftime('%Y')
month_str = today.strftime('%m')

# =============== Configurações ================
schema = 'workspace_db'
athena_table = 'tbl_coeres_painel_anomes_v1_gold'
hive_table = 'ghp00005.tbl_coeres_painel_anomes_v1_gold'
partition_cols = ['year', 'month', 'canal']  # todas devem ser string

# JDBC
jdbc_url = 'jdbc:hive2://<host>:<port>/<db>;transportMode=http;httpPath=cliservice'
jdbc_username = '<YOUR_USER>'
jdbc_password = '<YOUR_PASSWORD>'
jdbc_driver = 'HiveJDBC42.jar'
jdbc_jar_driver = 'com.cloudera.hive.jdbc.HS2Driver'
jdbc_jar_path = f"/path/to/{jdbc_driver}"

# =============== Leitura do Athena (Glue Catalog) ================
df = spark.read \
    .format("awsdatacatalog") \
    .option("catalog", "AwsDataCatalog") \
    .option("database", schema) \
    .option("table", athena_table) \
    .load()

# Filtro da partição desejada
df_partition = df.filter(
    (df["year"] == year_str) &
    (df["month"] == month_str)
)

# =============== Geração do CREATE TABLE dinâmico ================
def generate_create_table_sql(df, table_name, partition_columns):
    cols = []
    partitions = []

    for field in df.schema.fields:
        col_name = field.name
        data_type = field.dataType.simpleString()

        # Força partições como string
        if col_name in partition_columns:
            partitions.append(f"{col_name} STRING")
        else:
            # Conversão básica Spark -> Hive
            if data_type.startswith("string"):
                cols.append(f"{col_name} STRING")
            elif data_type.startswith("int"):
                cols.append(f"{col_name} INT")
            elif data_type.startswith("double"):
                cols.append(f"{col_name} DOUBLE")
            elif data_type.startswith("bigint"):
                cols.append(f"{col_name} BIGINT")
            elif data_type.startswith("timestamp"):
                cols.append(f"{col_name} TIMESTAMP")
            else:
                cols.append(f"{col_name} STRING")  # default

    create_stmt = f"CREATE TABLE {table_name} (\n  " + ",\n  ".join(cols) + "\n)"
    if partitions:
        create_stmt += f"\nPARTITIONED BY (\n  " + ",\n  ".join(partitions) + "\n)"
    create_stmt += "\nSTORED AS PARQUET"

    return create_stmt

# =============== Verifica se tabela existe e cria se necessário ================
def ensure_table_exists(jdbc_url, user, password, driver, jar_path, table_name, create_sql):
    conn = jaydebeapi.connect(driver, jdbc_url, [user, password], jar_path)
    cursor = conn.cursor()

    cursor.execute(f"SHOW TABLES LIKE '{table_name.split('.')[-1]}'")
    result = cursor.fetchall()

    if not result:
        print("Tabela não existe. Criando...")
        print(f"CREATE SQL:\n{create_sql}")
        cursor.execute(create_sql)
        print("Tabela criada com sucesso.")
    else:
        print("Tabela já existe. Continuando...")

    conn.close()

# =============== Deleta partições por canal ================
def delete_partition(year, month, canal):
    print(f"Deletando partição: year={year}, month={month}, canal={canal}")
    conn = jaydebeapi.connect(jdbc_jar_driver, jdbc_url, [jdbc_username, jdbc_password], jdbc_jar_path)
    cursor = conn.cursor()
    delete_sql = f"""
        DELETE FROM {hive_table}
        WHERE year = '{year}' AND month = '{month}' AND canal = '{canal}'
    """
    cursor.execute(delete_sql)
    conn.close()
    print("Partição deletada com sucesso.")

# Criação da tabela se não existir
create_sql = generate_create_table_sql(df_partition, hive_table, partition_cols)
ensure_table_exists(jdbc_url, jdbc_username, jdbc_password, jdbc_jar_driver, jdbc_jar_path, hive_table, create_sql)

# Identifica canais a serem sobrescritos
canais = [row["canal"] for row in df_partition.select("canal").distinct().collect()]
for canal in canais:
    delete_partition(year_str, month_str, canal)

# Reparticiona para melhor paralelismo
df_partition = df_partition.repartition(50, "canal")

# Escrita via JDBC
print("Escrevendo dados no Hadoop via JDBC...")
df_partition.write \
    .format("jdbc") \
    .option("url", jdbc_url) \
    .option("dbtable", hive_table) \
    .option("user", jdbc_username) \
    .option("password", jdbc_password) \
    .option("driver", jdbc_jar_driver) \
    .option("batchsize", 10000) \
    .option("numPartitions", 10) \
    .option("partitionColumn", "canal") \
    .mode("append") \
    .save()

print("Processo finalizado com sucesso.")