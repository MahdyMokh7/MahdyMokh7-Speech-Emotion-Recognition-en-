Use ser_db;

SHOW TABLES;

DESCRIBE Features;


Select * 
from features
limit 8;


SELECT file_name, bandwidth
FROM Features
WHERE bandwidth > 100
order by bandwidth asc;


SELECT emotion_category, emotion
FROM Features
GROUP BY emotion_category, emotion;

Select emotion_category, count(*) as Num_Audios
From Features
Group by emotion_category;

Select Count(*) as Total_Num_audios
From Features;


SELECT emotion_category, AVG(rms) AS avg_rms
FROM Features
GROUP BY emotion_category
ORDER BY avg_rms DESC;

SELECT emotion_category, AVG(zcr) AS avg_zcr
FROM Features
GROUP BY emotion_category
ORDER BY avg_zcr DESC;


Select emotion_category, Avg(chroma10) as avg_chroma10
From Features
Group by emotion_category
Having emotion_category = "SAD" or emotion_category = "ANG"
Order by Avg(chroma10) desc;
