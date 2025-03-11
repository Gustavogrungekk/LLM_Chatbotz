from pyspark.sql import SparkSession
from awsglue.utils import getResolvedOptions
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta
import jaydebeapi
import sys

# =============== Spark Session ================
spark = SparkSession.builder \
    .config('spark.sql.legacy.parquet.datetimeRebaseModeInRead', 'CORRECTED') \
    .config('spark.sql.files.maxPartitionBytes', 134217728) \
    .enableHiveSupport() \
    .getOrCreate()

# =============== Parâmetros do job ================
args = getResolvedOptions(sys.argv, ['MES'])

# Cálculo da partição alvo
today = (date.today() - timedelta(days=1) + relativedelta(months=-int(args['MES'])))
year = today.year
month = today.month

# =============== Configurações do Glue / Athena / JDBC ================
schema = 'workspace_db'
athena_table = 'tbl_coeres_painel_anomes_v1_gold'
hive_table = 'ghp00005.tbl_coeres_painel_anomes_v1_gold'
partition_column_year = 'year'
partition_column_month = 'month'
partition_column_canal = 'canal'

# JDBC (credenciais escondidas)
jdbc_url = 'jdbc:hive2://<host>:<port>/<db>;transportMode=http;httpPath=cliservice'
jdbc_username = '<YOUR_USER>'
jdbc_password = '<YOUR_PASSWORD>'
jdbc_driver = 'HiveJDBC42.jar'
jdbc_jar_driver = 'com.cloudera.hive.jdbc.HS2Driver'
jdbc_jar_path = f"/path/to/{jdbc_driver}"  # ajuste o caminho se necessário

# =============== Leitura da tabela do Glue Catalog (Athena) ================
df = spark.read \
    .format("awsdatacatalog") \
    .option("catalog", "AwsDataCatalog") \
    .option("database", schema) \
    .option("table", athena_table) \
    .load()

# Filtra partição desejada
df_partition = df.filter(
    (df[partition_column_year] == year) &
    (df[partition_column_month] == month)
)

# Coleta canais presentes na partição
canais = [row[partition_column_canal] for row in df_partition.select(partition_column_canal).distinct().collect()]

# =============== Função para deletar partição no Hadoop via JDBC ================
def delete_partition(year, month, canal):
    print(f"Deletando partição: year={year}, month={month}, canal={canal}")
    conn = jaydebeapi.connect(
        jdbc_jar_driver,
        jdbc_url,
        [jdbc_username, jdbc_password],
        jdbc_jar_path
    )
    cursor = conn.cursor()
    delete_sql = f"""
        DELETE FROM {hive_table}
        WHERE {partition_column_year} = {year}
          AND {partition_column_month} = {month}
          AND {partition_column_canal} = '{canal}'
    """
    cursor.execute(delete_sql)
    conn.close()
    print(f"Partição canal={canal} deletada com sucesso.")

# Deleta partições por canal
for canal in canais:
    delete_partition(year, month, canal)

# =============== Escrita via JDBC com paralelismo ================
df_partition = df_partition.repartition(50, partition_column_canal)

print(f"Gravando partições year={year}, month={month}...")
df_partition.write \
    .format("jdbc") \
    .option("url", jdbc_url) \
    .option("dbtable", hive_table) \
    .option("user", jdbc_username) \
    .option("password", jdbc_password) \
    .option("driver", jdbc_jar_driver) \
    .option("batchsize", 10000) \
    .option("numPartitions", 10) \
    .option("partitionColumn", partition_column_canal) \
    .mode("append") \
    .save()

print("Carga finalizada com sucesso.")