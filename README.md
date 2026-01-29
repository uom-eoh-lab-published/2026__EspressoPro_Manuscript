# Project information:

## General Description:

**First Author:** [Kristian Gurashi]
**Correspondence:** [Corresponding Author]
**Focus:** Single-cell surface protein expression analysis and Cell type annotation  
**Keywords:** Tapestri, ADT, Annotation, ML, Classifiers. 

## Publication link:

[Publication DOI/Link - To be updated upon publication]

## Content:

This is the code repository for the **EspressoPro** manuscript - a machine learning pipeline for automated cell type annotation of single-cell protein expression data (ADT). Here we described the logical steps towards the making of this tool which is primarily tailed for use on MissionBio Tapestri DNA+ADT data on blood or bone marrow derived monuclear cells.

**Related Pipeline Repository:** [uom-eoh-lab-published/2024__EspressoPro](https://github.com/uom-eoh-lab-published/2024__EspressoPro)

## Instructions on reproducibility:

1. **Clone the GitHub repository:**

```bash
git clone https://github.com/uom-eoh-lab-published/2024__EspressoPro_Manuscript.git
cd 2024__EspressoPro_Manuscript
```

2. **Set up the computational environment:**

```bash
# Create conda environment from provided file
conda env create -f Environments/Installation/Espressopro_manuscript.yml
conda activate Espressopro_manuscript

# Install the main EspressoPro pipeline
pip install git+https://github.com/uom-eoh-lab-published/2024__EspressoPro.git
```

3. **Execute analysis scripts in order:**

```bash
# 1. Download and organize datasets (R)
Rscript -e "rmarkdown::render('Scripts/01_Datasets_download.Rmd')"

# 2. Process and harmonize reference data (Python)
jupyter notebook Scripts/02_References_harmonisation.ipynb

# 3. Partition data for training/testing (R)
Rscript -e "rmarkdown::render('Scripts/03_Partitioning.Rmd')"

# 4. Train classification models (Python)
jupyter notebook Scripts/04_Models_training.ipynb

# 5. Map cell type ontologies (Python)
jupyter notebook Scripts/05_Ontologies.ipynb

# 6. Generate predictions and validate models (Python)
jupyter notebook Scripts/06_Models_prediction.ipynb
```

**Coded in/requires:**

![R Badge](https://img.shields.io/badge/R-v4.4.3-blue?style=flat&logo=R)
![Python Badge](https://img.shields.io/badge/Python-v3.10.18-green?style=flat&logo=Python). 

# Citation:

If using any code here published and/or data from this publication please remember to cite us:

**• BibTeX Style:**

```bibtex
@article{espressopro2024,
  title={EspressoPro: An Automated Machine Learning Driven Protein Annotator For Tapestri Data},
  author={[Author Names]},
  journal={[Journal Name]},
  year={2024},
  doi={[DOI]}
}
```

**• APA Style:**

> [Author Names]. (2024). EspressoPro: An Automated Machine Learning Driven Protein Annotator For Tapestri Data. *[Journal Name]*, [Volume(Issue)], [Pages]. https://doi.org/[DOI]

**• Vancouver Style:**

> [Author Names]. EspressoPro: An Automated Machine Learning Driven Protein Annotator For Tapestri Data. [Journal Name]. 2024;[Volume]:[Pages]. doi:[DOI]

# Networking:

If you wish to reach out you can connect with [First Author] at 👇:

[![LinkedIn](https://img.shields.io/badge/LinkedIn-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/[profile]) [![Twitter](https://img.shields.io/badge/Twitter-lightblue?style=flat&logo=twitter)](https://twitter.com/[handle]) [![Email](https://img.shields.io/badge/Email-red?style=flat&logo=gmail)](mailto:[email@domain.com])

# Biography:

To know more about [First Author Name] explore their biography and list of publications at 👇:

[![GitHub](https://img.shields.io/badge/GitHub-black?style=flat&logo=github)](https://github.com/[username]) [![ORCID](https://img.shields.io/badge/ORCID-green?style=flat&logo=orcid)](https://orcid.org/[orcid-id]) [![Google Scholar](https://img.shields.io/badge/Google%20Scholar-blue?style=flat&logo=google-scholar)](https://scholar.google.com/citations?user=[user-id])
