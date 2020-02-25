ALTER TABLE apple_data RENAME "Tweet Id" to tweet_id;
ALTER TABLE apple_data RENAME "Tweet content" to tweet_content;
ALTER TABLE apple_data RENAME "Is a RT" to retweeted;

ALTER TABLE apple_data ALTER COLUMN "Date" TYPE Date USING to_date("Date", 'YYYY-MM-DD');

DROP VIEW IF EXISTS prepared_data_5;
CREATE VIEW prepared_data_5 AS 
    SELECT t.* FROM (
        SELECT 
            row_number() OVER(ORDER BY ad.index ASC) AS row,
            ap_t."Close",
            ap_t."Open",
            ap_t."Low",
            ap_t."High",
            ap_t."Volume",
            ap_t."Date" as "Date",

            ad.tweet_content as "text"

        FROM apple_data AS ad
        INNER JOIN "AAPL_ticker" AS ap_t ON ad."Date" = ap_t."Date"
        ORDER BY "Date" ASC
    ) as t
    WHERE t.row % 5 = 0;


DROP VIEW IF EXISTS prepared_data_4;
CREATE VIEW prepared_data_4 AS 
    SELECT t.* FROM (
        SELECT 
            row_number() OVER(ORDER BY ad.index ASC) AS row,
            ap_t."Close",
            ap_t."Open",
            ap_t."Low",
            ap_t."High",
            ap_t."Volume",
            ap_t."Date" as "Date",

            ad.tweet_content as "text"

        FROM apple_data AS ad
        INNER JOIN "AAPL_ticker" AS ap_t ON ad."Date" = ap_t."Date"
        ORDER BY "Date" ASC
    ) as t
    WHERE t.row % 4 = 0;

DROP VIEW IF EXISTS prepared_data_3;
CREATE VIEW prepared_data_3 AS 
    SELECT t.* FROM (
        SELECT 
            row_number() OVER(ORDER BY ad.index ASC) AS row,
            ap_t."Close",
            ap_t."Open",
            ap_t."Low",
            ap_t."High",
            ap_t."Volume",
            ap_t."Date" as "Date",
            
            ad.tweet_content as "text"

        FROM apple_data AS ad
        INNER JOIN "AAPL_ticker" AS ap_t ON ad."Date" = ap_t."Date"
        ORDER BY "Date" ASC
    ) as t
    WHERE t.row % 3 = 0;

DROP VIEW IF EXISTS prepared_data;
CREATE VIEW prepared_data AS 
    SELECT t.* FROM (
        SELECT 
            row_number() OVER(ORDER BY ad.index ASC) AS row,
            ap_t."Close",
            ap_t."Open",
            ap_t."Low",
            ap_t."High",
            ap_t."Volume",
            ap_t."Date" as "Date",

            ad.tweet_content as "text"

        FROM apple_data AS ad
        INNER JOIN "AAPL_ticker" AS ap_t ON ad."Date" = ap_t."Date"
        ORDER BY "Date" ASC
    ) as t;

GRANT ALL ON prepared_data_3 TO stock;
GRANT ALL ON prepared_data_4 TO stock;
GRANT ALL ON prepared_data_5 TO stock;
GRANT ALL ON prepared_data TO stock;