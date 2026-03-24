from pyspark.sql import SparkSession
from pyspark.sql.functions import col

spark = SparkSession.builder.appName("KPMG_Test").getOrCreate()

sales_df = spark.read.csv("/samples/sales.csv", header=True, inferSchema=True)
customers_df = spark.read.csv("/samples/customers.csv", header=True, inferSchema=True)
products_df = spark.read.csv("/samples/products.csv", header=True, inferSchema=True)

sales_df.createOrReplaceTempView("sales")
customers_df.createOrReplaceTempView("customers")
products_df.createOrReplaceTempView("products")
sales_df.printSchema()
sales_df.show(200)
customers_df.printSchema()
customers_df.show(5)
products_df.printSchema()
products_df.show(5)

#Step 1: Using SQL, perform joins between Sales, Customer, and Product on their respective IDs
#(lookup in Sales table based on customer_id from Customer to get customer details, 
#and lookup in Sales table based on product_id from Product to get product details). 
#Find the total purchases by a customer. Write it to a view.

result_df =spark.sql("""
Select SUM(s.total_amount) AS total_amount,
  s.sale_id,customers.customer_id, customers.first_name , products.product_name 
From sales s
LEFT JOIN customers
ON s.customer_id = customers.customer_id 
LEFT JOIN  products
ON s.product_id + 100 = products.product_id
GROUP BY 
  s.sale_id,customers.customer_id, customers.first_name , products.product_name
""")

display(result_df)

#Find the duplicate Values for tables sales.products and customers
#Find the null values for the Total_amount in Sales table. 
# verify sales date should not be future date
# verify product price should not be 0
#sales quantity should not be other than integer

#Identify whose product sales have increased over the years using SQL or PySpark. 


#For last question my approach would be to select customer id and details from cutomer table who is associated with sales id and the quanittity , we will compare the quantity from the sale of 2025 with 2026 (if dataavailable) 


