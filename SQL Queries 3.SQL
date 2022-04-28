-- NUMERIC TYPES
-- negative scale
SELECT CAST (2345944.3123 AS NUMBER(10,-3)) FROM dual;

-- excess scale is rounded off
SELECT CAST (2345944.3123 AS NUMBER(10,2)) FROM dual;
-- but excess precision gives an error (did yesterday)

-- DATE Literals
-- 
SELECT DATE '1997-13-01' FROM dual;

-- TMESTAMP literals
select CAST (TIMESTAMP '2021-01-13 10:40:50.09' AS DATE) from dual;

-- use the same EXTRACT function with TIMESTAMP
select EXTRACT (YEAR FROM TIMESTAMP '2021-01-13 10:40:50.09') from dual;

-- Intervals
SELECT sysdate  - DATE '2021-12-01' FROM dual;

-- to figure out what date format is default
SELECT *
FROM nls_session_parameters
WHERE PARAMETER = 'NLS_DATE_FORMAT';

-- to set date format (valid for duration of session only)
ALTER SESSION SET NLS_DATE_FORMAT = 'YYYY MM DD';

-- how many of the 10 products with the highest markup have been sold
-- now with JOIN
SELECT 
    p.product_name,
    p.list_price - p.standard_cost AS markup,
    SUM(i.quantity),
    AVG(i.quantity),
    COUNT(i.product_id)
FROM 
    ot.products p
LEFT JOIN 
    ot.order_items i
	ON 
		i.product_id = p.product_id
LEFT JOIN 
    ot.orders o
	ON 
		i.order_id = o.order_id
WHERE 
    o.status <> 'Canceled'
	OR  o.order_id IS NULL
GROUP BY 
    p.product_id, 
    p.product_name, 
    p.list_price - p.standard_cost
ORDER BY 
    markup DESC
FETCH 
    FIRST 10 rows only;

-- More joins

-- list all warehouses with their region, country, city and state
SELECT 
    r.region_name, c.country_name, l.city, l.state, w.warehouse_name
FROM 
    ot.regions r
INNER JOIN ot.countries c
	ON  r.region_id = c.region_id
INNER JOINot.locations l
	ON l.country_id = c.country_id
INNER JOIN ot.warehouses w
	ON w.location_id = l.location_id;

-- List all countries without warehouses (see also subquery)
-- first, look at all combinations:
SELECT 
    c.country_name, 
    l.city, 
    l.state, 
    w.warehouse_name 
FROM ot.countries c
LEFT OUTER JOIN Ot.locations l 
   ON l.country_id = c.country_id
LEFT OUTER JOIN 
    ot.warehouses w 
   ON l.location_id = w.location_id
ORDER BY 1;

-- then build the query
SELECT 
    c.country_name
FROM ot.countries c
LEFT OUTER JOIN ot.locations l 
   ON l.country_id = c.country_id
LEFT OUTER JOIN ot.warehouses w 
   ON l.location_id = w.location_id
GROUP BY c.country_name
HAVING COUNT(w.location_id) = 0;

-- Self-join: list all employees and their direct manager
SELECT
    e.first_name,
    e.last_name,
    e.job_title,
    m.first_name   AS "manager first name",
    m.last_name    AS "manager last name",
    m.job_title    AS "manager job title"
FROM
    ot.employees   e
    INNER JOIN ot.employees   m ON e.manager_id = m.employee_id;

-- list salesmen in order of orders placed, and their manager
SELECT 
    e.first_name, 
    e.last_name, 
    e.employee_id,
    e.job_title,
    count(o.order_id)
FROM 
    ot.employees e 
INNER JOIN 
    ot.orders o 
   ON e.employee_id = o.salesman_id
GROUP BY e.employee_id,
    e.first_name, 
    e.last_name,
    e.job_title;

-- salesmen, sales, and their manager
SELECT 
    e.first_name, 
    e.last_name, 
    e.employee_id,
    e.job_title,
    m.first_name AS "Manager Name",
    m.last_name AS "Manager Last Name",
    m.job_title AS "Manager Title",
    SUM(o.order_id)
FROM 
    ot.employees e 
INNER JOIN 
    ot.orders o 
   ON e.employee_id = o.salesman_id
INNER JOIN 
    ot.order_items i 
   ON o.order_id = i.order_id
INNER JOIN
    ot.employees m
   ON e.manager_id = m.employee_id
GROUP BY e.employee_id,
    e.first_name, 
    e.last_name,
    e.job_title,
    m.first_name,
    m.last_name,
    m.job_title;

-- average value of sales ORDER
SELECT 
    SUM(i.quantity*i.unit_price)/COUNT(DISTINCT order_id)
FROM ot.order_items i;
-- list all orders above average VALUE -> see in SUBQUERIES


-- what are the total profits per product ?
SELECT 
    p.product_id,
    p.product_name,
    sum((p.list_price - p.standard_cost)*i.quantity) as profitx,
--    TO_CHAR(sum((p.list_price - p.standard_cost)*i.quantity), '9G999G999D99') as profit,
    count(i.product_id)
FROM 
    ot.order_items i
INNER JOIN ot.products p
  ON i.product_id = p.product_id
GROUP BY 
    p.product_id,
    p.product_name
ORDER BY profit DESC;

-- same as above, show profit sum along with lost (canceled) sales per product
WITH prep AS 
	 (SELECT 
		i.product_id,
		CASE 
		   WHEN o.status = 'Shipped' THEN i.quantity*(p.list_price - p.standard_cost)
		END as profit,
		CASE o.status
		   WHEN 'Canceled' THEN i.quantity*(p.list_price - p.standard_cost)
		END as lost
	 FROM ot.order_items i
		INNER JOIN ot.orders o
		  ON o.order_id = i.order_id
		INNER JOIN ot.products p
		 ON p.product_id = i.product_id
	 ) 
SELECT product_id,
        SUM(profit),
        SUM(lost)
FROM prep
group by product_id
HAVING sum(profit) is not null OR sum(lost) is not null
ORDER BY 2 DESC NULLS LAST, 3 DESC NULLS LAST;


-- list all orders above average VALUE -> see in SUBQUERIES
SELECT 
    i.order_id,
    sum(i.quantity*i.unit_price) as order_value
FROM ot.order_items i
GROUP BY i.order_id
HAVING sum(i.quantity*i.unit_price) > 
        (
        SELECT 
                SUM(i.quantity*i.unit_price)/COUNT(DISTINCT order_id)
        FROM ot.order_items i
        );


-- Subqueries
-- this will later be a CTE:
-- list all products with their profits (Shipped) as well as lost deals (Canceled)
SELECT product_id,
        SUM(profit),
        SUM(lost)
FROM 
 (SELECT 
    i.product_id,
    CASE 
       WHEN o.status = 'Shipped' THEN 
    i.quantity*(p.list_price - p.standard_cost)
    END as profit,
    CASE o.status
       WHEN 'Canceled' THEN 
    i.quantity*(p.list_price - p.standard_cost)
    END as lost
FROM ot.order_items i
INNER JOIN ot.orders o
  ON o.order_id = i.order_id
INNER JOIN ot.products p
 ON p.product_id = i.product_id
) 
group by product_id
HAVING sum(profit) is not null OR sum(lost) is not null
ORDER BY 2 DESC NULLS LAST, 3 DESC NULLS LAST;


-- Correlated subqueries
-- list all countries without locations
SELECT c.country_name
FROM ot.countries c
WHERE NOT EXISTS
        (SELECT 1
        FROM ot.locations l
        WHERE c.country_id = l.country_id
       );
   
-- list all countries without warehouses
SELECT c.country_name
FROM ot.countries c
WHERE NOT EXISTS
        (SELECT 1
        FROM ot.locations l
        WHERE c.country_id = l.country_id
        AND EXISTS
                (SELECT 1 
                FROM ot.warehouses w
                WHERE w.location_id = l.location_id
                )
        );

-- CTE
WITH prep AS 
 (SELECT 
    i.product_id,
    CASE 
       WHEN o.status = 'Shipped' THEN 
    i.quantity*(p.list_price - p.standard_cost)
    END as profit,
    CASE o.status
       WHEN 'Canceled' THEN 
    i.quantity*(p.list_price - p.standard_cost)
    END as lost
FROM ot.order_items i
INNER JOIN ot.orders o
  ON o.order_id = i.order_id
INNER JOIN ot.products p
 ON p.product_id = i.product_id
) 
SELECT product_id,
        SUM(profit),
        SUM(lost)
FROM prep
group by product_id
HAVING sum(profit) is not null OR sum(lost) is not null
ORDER BY 2 DESC NULLS LAST, 3 DESC NULLS LAST;

-- Hierarchical Query (ANSI)
WITH  employee_hierarchy (
        employee_id,
		first_name,
		last_name,
		job_title,
		manager_id,
		path)
   AS (
	SELECT
		employee_id,
		first_name,
		last_name,
		job_title,
		manager_id,
		last_name AS path
	FROM
		ot.employees
	WHERE
		manager_id IS NULL
	UNION ALL
	SELECT 
		e.employee_id,
		e.first_name,
		e.last_name,
		e.job_title,
		e.manager_id,
		employee_hierarchy.path || '->' || e.last_name
	FROM ot.employees e, employee_hierarchy
	WHERE e.manager_id = employee_hierarchy.employee_id
	)
SELECT * FROM employee_hierarchy;

-- HIERARCHICAL queries (ORACLE)
SELECT
    first_name,
    last_name,
    job_title,
    manager_id,
    sys_connect_by_path(last_name, '->') AS path
FROM
    ot.employees
START WITH
    manager_id IS NULL
CONNECT BY
    PRIOR employee_id = manager_id;

-----------------
-- more queries
-----------------
-- list product that have a price greater than the average for the category
select p.category_id, p.product_id,p.product_name, p.list_price
FROM ot.products p
WHERE p.list_price > 
(
select avg(list_price)
FROM ot.products p1 
WHERE p.category_id = p1.category_id
GROUP BY category_id
);

-- what are the sales by customer and category
SELECT 
      x.name, 
      c.category_name, 
      sum(i.quantity*i.unit_price)
 FROM ot.orders o
 JOIN ot.customers x
   ON x.customer_id = o.customer_id
 JOIN ot.order_items i
   ON o.order_id = i.order_id
 JOIN ot.products p
   ON i.product_id = p.product_id
 JOIN ot.product_categories c
   ON c.category_id = p.category_id
WHERE o.status = 'Shipped'
GROUP BY x.name, c.category_name
ORDER BY x.name;


