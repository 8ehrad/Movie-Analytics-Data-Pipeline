# Movie Data Pipeline with NiFi, MySQL, and Grafana

## Overview

This project is an end-to-end data engineering pipeline that automates the extraction, transformation, storage and visualisation of movie-related data. The pipeline utilises **Apache NiFi** for data ingestion and processing, **MySQL** for structured storage, and **Grafana** for data visualisation.

The entire solution is **containerised with Docker** and can be easily deployed using **Docker Compose**.


## Tech Stack

- **Apache NiFi** - Data ingestion and ETL processing
- **MySQL** - Relational database for storing processed data
- **Grafana** - Interactive dashboards for visualising movie insights
- **Python** - Custom scripts for data transformation and analysis
- **Docker & Docker Compose** - Containerised deployment

## Project Workflow

1. **Data Ingestion**  
   - Apache NiFi fetches movie data from an input file (`titles.txt`).
   - Data flows through multiple NiFi processors for transformation.

2. **Data Storage**  
   - Transformed data is inserted into a MySQL database (`imdb`).
   - The `movies` table contains data on reviews, ratings, and box office sales.

3. **Data Visualization**  
   - Grafana dashboards display movie analytics.
   - Users can compare different movies based on key performance metrics such as Box Office sales, IMDB and Metascore ratings, and AI-driven sentiment analysis of user reviews.

## Setup Instructions

### Running the Project with Docker Compose

The entire solution is **bundled into a Docker container** and can be started with **Docker Compose**.

### Apache NiFi

1. Start NiFi and access it at:  
   **[https://localhost:8443/nifi/](https://localhost:8443/nifi/)**
2. Login credentials:  
   - **Username**: `scc413`  
   - **Password**: `abcdefghijklm`
3. Run all processors to start the data pipeline.
4. **Important:** After NiFi reads the input file (`./nifi/datasets/titles.txt`), manually stop the `GetTitles` processor to prevent re-reading every 300 seconds.

### MySQL

1. Access MySQL at:  
   **[http://localhost:8080/](http://localhost:8080/)**
2. Login credentials:  
   - **System**: `MySQL`  
   - **Server**: `mysql`  
   - **Username**: `root`  
   - **Password**: `example`
3. The database used is **`imdb`**.
4. Check the `movies` table for processed data, including reviews analysis, ratings, and box office sales.

### Grafana

1. Access Grafana at:  
   **[http://localhost:3000/](http://localhost:3000/)**
2. Login credentials:  
   - **Username**: `admin`  
   - **Password**: `admin`
3. Navigate to **Dashboards** → Click on **Movies** panel to view visualized movie insights.

## Features

- Fully automated **ETL pipeline** for movie data processing.
- **SQL database** for structured storage.
- **Interactive dashboards** for exploring movie trends.
- Scalable design using **NiFi for data orchestration**.
