-- Simple Queries: REGIONS
SELECT 
    *
FROM 
    ot.regions
;

-- column names
SELECT 
    region_id, region_name
FROM 
    ot.regions
;

-- ordering
SELECT 
    region_id, region_name
FROM 
    ot.regions
ORDER BY region_name ASC
;

-- FILTER (WHERE)
SELECT 
    region_id, region_name
FROM 
    ot.regions
WHERE region_name = 'Asia';
	
-- FILTER (WHERE)
SELECT 
	country_id,
	country_name,
	region_id
FROM ot.countries
WHERE country_name = 'Germany'
OR 	country_name = 'Brazil'
;

-- FILTER: Pattern matching: LIKE (using %,_ ,ESCAPE), 
SELECT 
	country_id,
	country_name,
	region_id
FROM ot.countries
WHERE country_name LIKE 'U%'	
;

-- FILTER: REGEXP_LIKE
SELECT 
	first_name, 
	last_name
FROM ot.employees
WHERE REGEXP_LIKE (first_name, '^So(ph|f)i(a|e)$');

-- more .. FILTER (WHERE)
-- Group comparison: ANY/SOME, ALL
SELECT *
FROM ot.countries	
WHERE region_id > ALL
  (SELECT region_id
  FROM ot.regions
  WHERE region_name IN ('Americas','Asia'))
