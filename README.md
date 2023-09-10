**🌟 State-of-the-Art Affective Norms Extrapolation with Transformers 🌟**

---

✨ **Description:**

Welcome to our cutting-edge research hub, where we redefine the boundaries of extrapolating affective norms across multiple languages. Using state-of-the-art transformer neural networks, we unlock the profound emotionality of words, serving as an indispensable resource for experimental stimuli selection and in-depth sentiment analysis.

**🔍 Highlights:**

- **🌐 Multi-Lingual Mastery:** We've taken a multi-faceted approach, extrapolating norms for an impressive lineup of languages including English, Polish, Dutch, German, French, and Spanish.

- **🧠 Advanced Transformer Architecture:** Our avant-garde transformer-based neural network architecture stands unmatched in semantic and emotional norms extrapolation, exuding unmatched precision and adaptability for each language.

- **📊 Benchmark Surpassing Results:** Not just setting but elevating the standards, our revolutionary method boasts an average improvement of Δr = 0.1 in correlations with human judgments, overshadowing previous approaches.

- **🛠️ Next-Gen Stimuli Selection:** Venturing into uncharted territory, we unveil a pioneering, unsupervised stimuli selection method, ensuring words of profound semantic resonance are handpicked.

- **💻 Researcher's Web Tool Access:** While our main emphasis remains on the transformative models, we also provide a web application tool for researchers, serving as a practical touchpoint for direct extrapolation.

---

** Join us on this revolutionary journey, where state-of-the-art technology meets linguistic depth, and together, let's reshape the world of sentiment analysis and emotion understanding in a multilingual paradigm. Your feedback is not just welcomed—it's essential as we tread on this path of innovation and excellence. 🎉🚀**

---

📝 Citation: Plisiecki, H., & Sobieszek, A. (2023). Affective Norms Extrapolation Using Transformer-based Neural   
Networks and Its Application to Experimental Stimuli Selection. Behavior Research Methods.  
(accepted for publication)

---

**📂 Repository Structure & Description:**

- **`Root`**: Contains the primary scripts and utilities of the repository.
    - `datasets_prep.py`: Preparing and preprocessing datasets.
    - `dataset_and_model.py`: Dataset structures and model architecture.
    - `dutch.py`, `english.py`, `french.py`, `german.py`, `polish.py`, `spanish.py`: Language-specific training scripts.
    - `picking_words.py`: Picking words for robustness analyses.
    - `results.py`: Consolidates results from various models and analyses.
    - `stimuli_descent_bert.py`: Stimuli selection using a BERT-based descent algorithm.
    - `training_loop.py`: General training loop for neural networks.
    - `utils.py`: Utility functions and helpers.

- **`prediction_results`**: Stores prediction results from various models.
    - `abstractness_check_results.csv`: Results for the abstractness checks of words.
    - `abstractness_compare_results.csv`: Comparative analysis of abstractness.
    - Language-specific results: e.g., `dutch_results.csv`, `english_aoa_results.csv`.
    - `stimuli_descent_results.csv`: Results from stimuli descent using BERT.

- **`study_2_data`**: Secondary data for supplementary analyses.
    - `AOA_Kuperman.xlsx`: Dataset on Age of Acquisition in English language.
    - `Kazojc2009.txt`: Dataset for word frequency in Polish language.

- **`training_data`**: Training, validation, and testing datasets.
    - Language-specific datasets: e.g., `train_dutch.parquet`, `test_french.parquet`.
    - Warriner ANEW datasets: e.g., `warriner_anew_test.parquet`, `warriner_anew_train.parquet`.



