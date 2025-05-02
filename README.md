# Speech Emotion Recognition (SER) - Data Pipeline

Data analysis on human voices in English, and making a model to predict the emotion of the person talking based purely on its voice. We know that people communicate 50% through body language, 40% through tone of voice, and only 10% through the actual words they say. Our focus is on the 40% aspect, enabling our model to predict a person's emotion — this project aims to build a pipeline that extracts features from audio, preprocesses them, and stores them in a structured database. After all that we used machine learning for prediction of emotion for our Speech Emotion Recognition (SER) task.

## 📁 Project Structure

```
MAHDYMOKH7-SPEECH-EMOTION-RECOGNITION-EN/
│
├── .github/                          # GitHub Actions workflows
├── Analytics/                        # For storing csv analytics output
│
├── Database/                         # SQL scripts and DB schema visual
│   ├── Create_DataBase_all_features.sql
│   ├── Create_DataBase_final_features.sql
│   └── DB_Schema_Design_aggregated.svg
│
├── documentation/                   # Project documentation PDFs
│   ├── Phaze_1.pdf
│   └── Phaze_2.pdf
│
├── Helper/                          # Supporting files and code
│   ├── helperCodes/
│   ├── helperJunks/
│   ├── ImageStorytelling/
│   ├── sampleAudios/
│   ├── sampleWaveforms/
│   └── single sample audio_PreProcess_pipline_waveform/
│
├── NoteBooks/                       # Jupyter notebooks
│   ├── Feature Extraction ML.ipynb
│   ├── pre-process pipline CNN.ipynb
│   └── pre-process pipline ML.ipynb
│
├── scripts/                         # All scripts used in the pipeline
│   ├── database_connection.py
│   ├── feature_engineering.py
│   ├── feature_extraction.py
│   ├── import_to_db.py
│   ├── load_data.py
│   └── preprocess.py
│
├── StatisticalImages/              # (Optional) Folder for output plots or visualizations
│
├── .env                            # Environment variables (keep this secret!)
├── .gitignore                      # Files/folders to be ignored by Git
├── LICENSE                         # License file
├── pipeline.py                     # 🔁 Main script that runs the entire pipeline
├── requirements.txt                # Python dependencies
├── README.md                       # You’re reading this file
└── storytelling.pbix               # Power BI storytelling dashboard
```

## ⚙️ Requirements

Install dependencies before running:

```bash
pip install -r requirements.txt
```

---

## 🛠️ Setup Instructions

### 1. Configure MySQL

- Create a MySQL database and tables using the scripts in the `Database/` folder
- Ensure your MySQL service is running on `localhost:3306`
- Set up a `.env` file in the root directory to store your DB credentials:

```
MYSQL_USER=root
MYSQL_PASSWORD=your_password
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_DB=SER_DB
```

Make sure `.env` is added to `.gitignore`.

---

## 🚀 Run the Pipeline

Just run the main pipeline script:

```bash
python pipeline.py
```

It will automatically:
1. Preprocess raw audio
2. Extract features and save to CSV
3. Import features into MySQL
4. Load them back into pandas
5. Apply feature engineering
6. Store the final features in the database

---

## 📌 Notes

- You don’t need to run each script manually — `pipeline.py` automates everything.
- Ensure required folders like `Features/` or `../DataSet/` exist.
- If ffmpeg is missing (used by `pydub`), install it and add to PATH.

---

## ❗ Troubleshooting

- **Missing files or folders**: Verify paths and create any missing directories
- **MySQL connection issues**: Check if credentials are correct and MySQL is running
- **CSV not found**: Make sure intermediate CSV files are generated correctly

---

## 📜 License

This project is licensed under the MIT License.
