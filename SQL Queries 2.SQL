-----------------------
-- Operators precedence
-----------------------
-- This selects order 65, which < '2015-01-01'
SELECT
    *
FROM
    ot.orders
WHERE
    order_date < '01 JAN 2015'
OR  salesman_id < 60 
AND status = 'Canceled';

-- This does NOT select order 65, because status is not 'Canceled'
SELECT
    *
FROM
    ot.orders
WHERE
   ( order_date < '01 JAN 2015'
OR  salesman_id < 60 )
AND status = 'Canceled';

--------
-- CASE
--------

SELECT
    CASE
        WHEN job_title LIKE '%Manager' THEN 'Manager'
        WHEN job_title LIKE '%President' THEN 'President'
        ELSE 'Worker'
    END,
    job_title,
    first_name,
    last_name
FROM
    ot.employees
FETCH FIRST 10 ROWS ONLY;

-- CASTING
-- Casting to a lower definition type loses scale
SELECT CAST(14352.223 AS DECIMAL (10,2)) FROM dual;
SELECT CAST(14321.55 AS INT) FROM dual;

-- Casting to a fixed length string pads woth blank to the right, to the field length
SELECT 'Average: ' || CAST(14352.321 AS CHAR(21)) || ' <' FROM dual;


----------
-- JOINS
----------

SELECT *
FROM ot.warehouses w
JOIN ot.locations  l
ON w.location_id = l.location_id;

SELECT 
    *
FROM 
    ot.regions r
INNER JOIN 
    ot.countries c
ON 
    r.region_id = c.region_id;


-- Right join and filter: locations without warehouses
SELECT w.warehouse_name,
        l.city
FROM ot.locations l 
LEFT OUTER JOIN ot.warehouses w
ON w.location_id = l.location_id
WHERE w.location_id IS NULL;


------------------------------
-- Simple Aggregate Functions
------------------------------

SELECT AVG(i.quantity)
FROM ot.order_items i;

-- using CAST to truncate decimals
SELECT 
    CAST (AVG(i.quantity) AS NUMBER(8,2)) AS AVG_QTY
FROM 
    ot.order_items i;

-- average number of items for canceled orders 
-- casting to various types 
SELECT o.order_id, 
        avg(i.quantity) as avg_qty_canceled,
        CAST (avg(i.quantity) as INTEGER) as avg_qty_int,
        CAST (avg(i.quantity) as CHAR(40)) as avg_qty_char
FROM ot.order_items i
JOIN ot.orders o
ON o.order_id = i.order_id
WHERE o.status = 'Canceled'
GROUP BY (o.order_id);

-- other aggregations
SELECT o.order_id, 
        MIN(i.quantity),
        MAX(i.quantity),
        SUM(i.quantity)
FROM ot.order_items i
JOIN ot.orders o
ON o.order_id = i.order_id
WHERE o.status = 'Canceled'
GROUP BY (o.order_id);

-- average number of orders per year per salesman, sorted descending
SELECT 
        EXTRACT(YEAR from order_date),
        count(*),
        salesman_id
FROM ot.orders 
GROUP BY salesman_id,
        EXTRACT(YEAR from order_date)
ORDER BY 1, 2 DESC;

-- customers with more than 1 order pending
SELECT
    name as "Customer Name",
    c.customer_id,
    COUNT(o.order_id)
FROM
    ot.customers   c
    JOIN ot.orders      o ON o.customer_id = c.customer_id
          and o.status = 'Pending'
GROUP BY
    c.name,
    c.customer_id
HAVING
    COUNT(o.order_id) > 1
order BY
    c.customer_id;

-- how many of the 10 products with the highest markup have been sold
-- use commented out WHERE to exclude those items canceled
SELECT p.product_name,
        p.list_price - p.standard_cost as markup,
        sum(i.quantity),
        avg(i.quantity),
        count(i.product_id)
FROM ot.products p
LEFT JOIN ot.order_items i
ON i.product_id = p.product_id
-- WHERE i.order_id in (SELECT order_id FROM ot.orders WHERE status <> 'Canceled')
--        OR i.order_id IS NULL
GROUP BY p.product_id, p.product_name, p.list_price - p.standard_cost
ORDER BY markup DESC
FETCH FIRST 10 rows only;

