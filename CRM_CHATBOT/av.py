import boto3
from datetime import datetime
import awswrangler as wr
import pandas as pd

def save_to_athena(depara_table, s3_output_path, database_name, table_name):
    # Convert the depara_table list to a Pandas DataFrame
    df = pd.DataFrame(depara_table)
    
    # Save the DataFrame to S3 in Parquet format
    wr.s3.to_parquet(
        df=df,
        path=s3_output_path,
        dataset=True,
        mode="overwrite",
        database=database_name,
        table=table_name
    )
    print(f"Data saved to Athena table {table_name} in database {database_name}.")

# Inicializa o cliente do Glue (usado para consultar o catálogo do Athena) com uma região específica
glue_client = boto3.client('glue', region_name='sa-east-1')

def get_last_partitions(allowed_databases=None):
    # Lista todos os bancos de dados no catálogo do Glue
    databases = glue_client.get_databases()['DatabaseList']
    depara_table = []

    for db in databases:
        database_name = db['Name']
        
        # Skip databases not in the allowed list if specified
        if allowed_databases and database_name not in allowed_databases:
            continue
        
        # Paginação
        next_token = None
        tables = []
        while True:
            response = glue_client.get_tables(DatabaseName=database_name, NextToken=next_token) if next_token else glue_client.get_tables(DatabaseName=database_name)
            tables.extend(response['TableList'])
            next_token = response.get('NextToken')
            if not next_token:
                break
        
        for table in tables:
            table_name = table['Name']
            
            # Paginação para listar todas as partições da tabela
            next_token = None
            partitions = []
            try:
                while True:
                    response = glue_client.get_partitions(DatabaseName=database_name, TableName=table_name, NextToken=next_token) if next_token else glue_client.get_partitions(DatabaseName=database_name, TableName=table_name)
                    partitions.extend(response.get('Partitions', []))
                    next_token = response.get('NextToken')
                    if not next_token:
                        break
                
                if partitions:
                    # Obtém a última partição (assumindo que as partições estão ordenadas)
                    last_partition = sorted(partitions, key=lambda p: p['Values'])[-1]
                    partition_values = last_partition['Values']
                    partition_keys = [key['Name'] for key in table['PartitionKeys']]
                    
                    partition_conditions = " and ".join(
                        f"{key}='{value}'" 
                        for key, value in zip(partition_keys, partition_values)
                    )
                    
                    depara_table.append({
                        'database': database_name,
                        'table': table_name,
                        'partition': partition_conditions,
                        'updated_at': datetime.now().strftime("%Y-%m-%d")
                    })
            except Exception as e:
                print(f"Error getting partitions for {database_name}.{table_name}: {str(e)}")
                continue

    return depara_table

#========================================================================#
# Configurações para salvar os dados no Athena
s3_output_path = "s3://your-bucket-name/path/to/depara_table/"
athena_database_name = "your_athena_database"
athena_table_name = "depara_table"

# Define a lista de databases
allowed_databases = ['workspace']

# Chama a função para obter as partições e salva no Athena
depara = get_last_partitions(allowed_databases)
save_to_athena(depara, s3_output_path, athena_database_name, athena_table_name)
