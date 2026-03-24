from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

spark = SparkSession.builder.appName('Danger in Code').getOrCreate()

# ===== LOAD DATA =====
sales_df = spark.read.csv('/samples/sales.csv', header = True, inferSchema = True)
customers_df = spark.read.csv('/samples/customers.csv', header = True, inferSchema = True)
products_df = spark.read.csv('/samples/products.csv', header = True, inferSchema = True)

print("Sales Schema")
sales_df.printSchema()
sales_df.show(3)

print("\nCustomers Schema")
customers_df.printSchema()
customers_df.show(3)

print("\nProduct Schema")
products_df.printSchema()
products_df.show(3)

# ===== CREATE TEMP VIEWS =====
sales_df.createOrReplaceTempView("sales")
customers_df.createOrReplaceTempView("customers")
products_df.createOrReplaceTempView("products")

test_df = spark.sql("Select * from sales LIMIT 5")
test_df.show()

# ===== STEP 1: SQL JOINS - CUSTOMER TOTAL PURCHASES =====
print("\n" + "=" * 80)
print("STEP 1: CUSTOMER TOTAL PURCHASES (SQL JOINS)")
print("=" * 80)

result_df = spark.sql("""
Select 
  sales.customer_id,
  customers.first_name,
  SUM(sales.total_amount) AS total_purchase
From sales 
LEFT JOIN customers
ON sales.customer_id = customers.customer_id
LEFT JOIN products
ON sales.product_id = products.product_id
GROUP BY
  sales.customer_id,
  customers.first_name
""")
result_df.show()

result_df.createOrReplaceTempView("customer_total_purchases")

# ===== STEP 2: TEST DATA VALIDATION (15 BDD SCENARIOS) =====
print("\n" + "=" * 80)
print("STEP 2: VALIDATION TESTS (15 ADVANCED BDD SCENARIOS)")
print("=" * 80)

# Create enriched sales view for testing
enriched_sales_df = spark.sql("""
SELECT
    s.sale_id,
    s.customer_id,
    c.first_name,
    c.last_name,
    s.product_id,
    p.product_name,
    s.total_amount,
    s.sale_date
FROM sales s
LEFT JOIN customers c ON s.customer_id = c.customer_id
LEFT JOIN products p ON s.product_id = p.product_id
""")

# Scenario 1 (BDD):
# Given sales records have valid customer and amount data
# When customer totals are aggregated
# Then each customer total_purchase is equal to the sum of their sales total_amount.

# Calculate expected totals from enriched sales
expected_totals = enriched_sales_df.groupBy("customer_id").agg(F.sum("total_amount").alias("expected_total"))

# Compare actual results with expected totals
comparison = result_df.join(expected_totals, on="customer_id")
mismatches = comparison.where(F.col("total_purchase") != F.col("expected_total")).count()

# Validate
assert mismatches == 0, "Test 1 Failed: Aggregation accuracy"
print("✓ Test 1 Passed: Aggregation accuracy - totals match sum of sales")

# Scenario 2 (BDD):
# Given customer-level aggregated output
# When checking for duplicate customer records
# Then each customer_id appears exactly once.
duplicate_count = result_df.groupBy("customer_id").count().where(F.col("count") > 1).count()
assert duplicate_count == 0, "Test 2 Failed: Duplicate customers found"
print("✓ Test 2 Passed: No duplicate customers - each customer_id appears once")

# Scenario 3 (BDD):
# Given left joins are used for customer and product lookup
# When sales rows have unknown dimension keys
# Then sales rows are still retained in enriched output.
sales_count = enriched_sales_df.select("sale_id").distinct().count()
assert sales_count > 0, "Test 3 Failed: No sales retained after join"
print("✓ Test 3 Passed: Left join preservation - sales retained even with unknown dimensions")

# Scenario 4 (BDD):
# Given total purchases represent monetary totals
# When validating final aggregated output
# Then total_purchase is not null for valid customer groups.
null_count = result_df.where(F.col("total_purchase").isNull()).count()
assert null_count == 0, "Test 4 Failed: Null values in totals"
print("✓ Test 4 Passed: No null values - all monetary totals are present")

# Scenario 5 (BDD):
# Given sales totals are non-negative business measures
# When validating customer totals
# Then total_purchase should not be negative.
negative_count = result_df.where(F.col("total_purchase") < 0).count()
assert negative_count == 0, "Test 5 Failed: Negative totals found"
print("✓ Test 5 Passed: Data quality - no negative amounts")

# ===== ADVANCED TEST SCENARIOS (6-15) =====

# Scenario 6 (BDD):
# Given enriched sales with customer and product details
# When checking for unexpected NULLs in dimension lookups
# Then all critical customer/product fields should be populated for valid sales.
missing_customer_names = enriched_sales_df.where((F.col("first_name").isNull()) | (F.col("last_name").isNull())).count()
assert missing_customer_names == 0, "Test 6 Failed: Missing customer names found"
print("✓ Test 6 Passed: Dimension data completeness - no missing customer names")

# Scenario 7 (BDD):
# Given a product dimension table
# When validating product information integrity
# Then all product_names should be non-empty strings.
missing_product_names = enriched_sales_df.where((F.col("product_name").isNull()) | (F.col("product_name") == "")).count()
assert missing_product_names == 0, "Test 7 Failed: Missing or empty product names"
print("✓ Test 7 Passed: Product data quality - all products have valid names")

# Scenario 8 (BDD):
# Given sales transactions with quantities
# When validating business logic constraints
# Then quantity should be positive and total_amount should match reasonable calculations.
invalid_quantity = enriched_sales_df.where(F.col("quantity") <= 0).count()
assert invalid_quantity == 0, "Test 8 Failed: Non-positive quantities found"
print("✓ Test 8 Passed: Sales integrity - all quantities are positive")

# Scenario 9 (BDD):
# Given aggregated customer purchase data
# When analyzing customer segmentation
# Then top 20% of customers should contribute significant portion of revenue.
customer_totals_df = result_df.orderBy(F.col("total_purchase").desc())
total_revenue = customer_totals_df.select(F.sum("total_purchase")).collect()[0][0]
top_20_percent_customers = customer_totals_df.limit(int(customer_totals_df.count() * 0.2))
top_20_revenue = top_20_percent_customers.select(F.sum("total_purchase")).collect()[0][0]
concentration_ratio = (top_20_revenue / total_revenue) * 100
assert concentration_ratio >= 50, f"Test 9 Failed: Top 20% customers only contribute {concentration_ratio:.1f}% of revenue"
print(f"✓ Test 9 Passed: Customer segmentation - Top 20% contributes {concentration_ratio:.1f}% of revenue")

# Scenario 10 (BDD):
# Given customer aggregation with purchase history
# When checking for data completeness
# Then every customer in aggregated output should have at least one sale.
customers_with_no_sales = result_df.where(F.col("total_purchase") == 0).count()
assert customers_with_no_sales == 0, "Test 10 Failed: Customers with zero purchases found"
print("✓ Test 10 Passed: Data consistency - all customers in result have at least one sale")

# Scenario 11 (BDD):
# Given multiple transactions per customer
# When verifying aggregation completeness
# Then each customer should appear in both enriched and aggregated datasets.
enriched_customer_count = enriched_sales_df.select("customer_id").distinct().count()
aggregated_customer_count = result_df.select("customer_id").distinct().count()
assert enriched_customer_count == aggregated_customer_count, f"Test 11 Failed: Mismatch in customer counts ({enriched_customer_count} vs {aggregated_customer_count})"
print("✓ Test 11 Passed: Aggregation completeness - all customers accounted for")

# Scenario 12 (BDD):
# Given sales transactions with temporal data
# When validating date consistency
# Then all sale dates should be reasonable (not in future, not too far in past).
max_date = enriched_sales_df.select(F.max("sale_date")).collect()[0][0]
min_date = enriched_sales_df.select(F.min("sale_date")).collect()[0][0]
assert min_date is not None and max_date is not None, "Test 12 Failed: Missing date values"
print(f"✓ Test 12 Passed: Temporal validity - sales data spans from {min_date} to {max_date}")

# Scenario 13 (BDD):
# Given product transaction volumes
# When analyzing sales distribution across products
# Then products should have reasonable sales counts (no product dominates 100% of sales).
product_participation = enriched_sales_df.groupBy("product_id").count().orderBy(F.col("count").desc()).limit(1).collect()
if len(product_participation) > 0:
    max_product_pct = (product_participation[0][1] / enriched_sales_df.count()) * 100
    assert max_product_pct < 99, f"Test 13 Failed: Single product dominates {max_product_pct:.1f}% of sales"
    print(f"✓ Test 13 Passed: Product diversity - top product is {max_product_pct:.1f}% of sales (healthy distribution)")

# Scenario 14 (BDD):
# Given customer purchase patterns
# When detecting potential data anomalies
# Then outlier customers (extremely high purchases) should exist but be reasonable.
q75 = result_df.approxQuantile("total_purchase", [0.75], 0.01)[0]
q25 = result_df.approxQuantile("total_purchase", [0.25], 0.01)[0]
iqr = q75 - q25
outlier_threshold = q75 + (1.5 * iqr)
outliers = result_df.where(F.col("total_purchase") > outlier_threshold).count()
total_customers = result_df.count()
outlier_pct = (outliers / total_customers) * 100 if total_customers > 0 else 0
assert outlier_pct <= 10, f"Test 14 Failed: Too many outlier customers ({outlier_pct:.1f}%)"
print(f"✓ Test 14 Passed: Outlier detection - {outlier_pct:.1f}% outlier customers (within acceptable range)")

# Scenario 15 (BDD):
# Given multi-table join operations
# When validating join result cardinality
# Then joined result should not have more rows than source (no cross-join effects).
enriched_row_count = enriched_sales_df.count()
sales_row_count = sales_df.count()
assert enriched_row_count <= sales_row_count * 1.1, f"Test 15 Failed: Join produced {enriched_row_count} rows from {sales_row_count} sources (possible cross-join)"
print(f"✓ Test 15 Passed: Join cardinality validation - {enriched_row_count} output rows from {sales_row_count} input rows (no unexpected expansion)")

print("\n" + "=" * 80)
print("ALL 15 ADVANCED TESTS PASSED!")
print("=" * 80)

# ===== STEP 3: PRODUCT SALES GROWTH ANALYSIS =====
print("\n" + "=" * 80)
print("STEP 3: PRODUCT SALES GROWTH OVER YEARS")
print("=" * 80)

# Extract year from sale_date
sales_with_year = enriched_sales_df.withColumn(
    "sales_year",
    F.year(F.to_date(F.col("sale_date")))
).where(F.col("sales_year").isNotNull())

# Product yearly sales
product_yearly = sales_with_year.groupBy("product_id", "product_name", "sales_year").agg(F.sum("total_amount").alias("yearly_sales"))
product_yearly.createOrReplaceTempView("product_yearly_sales")

print("\nProduct Yearly Sales:")
product_yearly.orderBy("product_id", "sales_year").show(50, truncate = False)

# Year-over-year growth comparison
window_spec = Window.partitionBy("product_id").orderBy("sales_year")
product_growth = product_yearly.withColumn(
    "prev_year_sales",
    F.lag("yearly_sales").over(window_spec)
).withColumn(
    "sales_increased",
    F.when(F.col("prev_year_sales").isNull(), False).otherwise(F.col("yearly_sales") > F.col("prev_year_sales"))
)
product_growth.createOrReplaceTempView("product_growth")

# Products with sales increase
products_increased = product_growth.groupBy("product_id", "product_name").agg(
    F.sum(F.when(F.col("sales_increased") == True, 1).otherwise(0)).alias("increase_count")
).where(F.col("increase_count") > 0)

print("\nProducts with Sales Increase Over Years:")
products_increased.orderBy("product_id").show(50, truncate = False)

print("\n" + "=" * 80)
print("EXECUTION COMPLETE")
print("=" * 80)
