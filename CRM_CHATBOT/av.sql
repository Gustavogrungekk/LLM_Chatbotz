WITH clientes_unicos AS (
    SELECT 
        cliente_id,
        MIN(canal) AS canal,  -- Define um único canal para cada cliente
        MIN(produto) AS produto,  -- Define um único produto para cada cliente
        MAX(CASE WHEN etapa = 'Contratacao' THEN 1 ELSE 0 END) AS contratou,
        MAX(CASE WHEN etapa = 'Potencial' THEN 1 ELSE 0 END) AS potencial
    FROM tabela_funnel
    GROUP BY cliente_id
)
SELECT 
    canal, 
    produto,
    COUNT(DISTINCT CASE WHEN contratou = 1 THEN cliente_id END) AS clientes_contratacao,
    COUNT(DISTINCT CASE WHEN potencial = 1 THEN cliente_id END) AS clientes_potenciais,
    CASE 
        WHEN COUNT(DISTINCT CASE WHEN potencial = 1 THEN cliente_id END) = 0 
        THEN 0 -- Evita divisão por zero
        ELSE 
            COUNT(DISTINCT CASE WHEN contratou = 1 THEN cliente_id END) * 100.0 / 
            COUNT(DISTINCT CASE WHEN potencial = 1 THEN cliente_id END) 
    END AS eficiencia
FROM clientes_unicos
GROUP BY canal, produto;
