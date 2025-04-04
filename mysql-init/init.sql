CREATE DATABASE imdb;

USE imdb;

DROP TABLE IF EXISTS movies;

CREATE TABLE movies (
   id INT AUTO_INCREMENT PRIMARY KEY, 
   title VARCHAR(255) UNIQUE NOT NULL,
   positive INT NOT NULL,
   negative INT NOT NULL, 
   neutral INT NOT NULL,
   totalReviews INT,
   imdb FLOAT, 
   metascore INT,
   cleanedBoxOffice INT
);
