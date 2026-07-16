# 🌿 Plant Leaf Classification using Deep Learning

An AI-powered web application that classifies **medicinal plant leaves** from uploaded images using a custom-built **Convolutional Neural Network (CNN)**. Along with predicting the plant species, the application leverages the **Groq LLM** to provide detailed information about the identified plant, including its medicinal uses, botanical description, and other relevant insights.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red?style=for-the-badge&logo=pytorch)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green?style=for-the-badge&logo=fastapi)
![HTML](https://img.shields.io/badge/Frontend-HTML5%20%7C%20CSS3%20%7C%20JavaScript-orange?style=for-the-badge&logo=html5)
![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)

---

## 📌 Overview

Identifying medicinal plants manually requires botanical knowledge and can often lead to misclassification. This project provides an AI-based solution capable of recognizing plant species directly from leaf images.

The system combines:

- 🧠 Deep Learning for image classification
- ⚡ FastAPI for high-performance backend
- 🤖 Groq LLM for AI-generated plant information
- 🎨 Responsive frontend using HTML and CSS 

---

## ✨ Features

- 🌿 Classifies **12 medicinal plant species**
- 📷 Upload leaf images directly from your browser
- ⚡ Fast and accurate predictions
- 🤖 AI-generated information using Groq LLM
- 📊 Displays predicted plant class
- 🎨 Modern responsive UI
- 🚀 REST API built with FastAPI
- 🔥 Easy to extend with additional plant classes

---

# 📸 Application Workflow

```text
              Upload Leaf Image
                      │
                      ▼
            Image Preprocessing
                      │
                      ▼
         Custom CNN Classification Model
                      │
                      ▼
           Predicted Plant Species
                      │
                      ▼
              Groq Large Language Model
                      │
                      ▼
      Plant Description & Medicinal Uses
```

---

# 🛠️ Tech Stack

## Backend

- FastAPI
- Python
- PyTorch
- Pillow
- LangChain
- Groq API

## Frontend

- HTML5
- CSS3
- Jinja2 Templates

## Deep Learning

- Convolutional Neural Network (CNN)
- TorchVision
- Image Transformations

---

# 🧠 CNN Architecture

The plant classifier is implemented using a **custom Convolutional Neural Network (CNN)** developed entirely in **PyTorch**.

The network consists of:

- 3 Convolutional Layers
- ReLU Activation
- Max Pooling after each convolution block
- 4 Fully Connected Layers
- Final output layer containing **12 plant classes**

---

## Network Architecture

| Layer | Configuration | Output |
|---------|--------------|---------|
| Input | RGB Image | 224 × 224 × 3 |
| Conv2D-1 | 3 → 12, Kernel 3×3 | Feature Maps |
| ReLU | Activation | |
| MaxPool | 3×3, Stride 3 | |
| Conv2D-2 | 12 → 24, Kernel 3×3 | |
| ReLU | Activation | |
| MaxPool | 3×3 | |
| Conv2D-3 | 24 → 28, Kernel 3×3 | |
| ReLU | Activation | |
| MaxPool | 3×3 | |
| Flatten | 28×7×7 | 1372 Features |
| Linear | 1372 → 1513 | |
| Linear | 1513 → 800 | |
| Linear | 800 → 400 | |
| Output | 400 → 12 | Plant Class |

---

## CNN Flow

```text
Input Image (224×224×3)
          │
          ▼
Conv2D (3 → 12)
          │
        ReLU
          │
     MaxPooling
          │
          ▼
Conv2D (12 → 24)
          │
        ReLU
          │
     MaxPooling
          │
          ▼
Conv2D (24 → 28)
          │
        ReLU
          │
     MaxPooling
          │
          ▼
Flatten
          │
          ▼
FC (1372 → 1513)
          │
        ReLU
          │
          ▼
FC (1513 → 800)
          │
        ReLU
          │
          ▼
FC (800 → 400)
          │
        ReLU
          │
          ▼
FC (400 → 12)
          │
          ▼
Predicted Plant Class
```

---

## PyTorch Implementation

```python
class CNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 12, kernel_size=3)
        self.conv2 = nn.Conv2d(12, 24, kernel_size=3)
        self.conv3 = nn.Conv2d(24, 28, kernel_size=3)

        self.pool = nn.MaxPool2d(kernel_size=3, stride=3)
        self.relu = nn.ReLU(inplace=True)

        self.fc1 = nn.Linear(28 * 7 * 7, 1513)
        self.fc2 = nn.Linear(1513, 800)
        self.fc3 = nn.Linear(800, 400)
        self.fc4 = nn.Linear(400, 12)

    def forward(self, x):

        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))

        x = torch.flatten(x, 1)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))

        x = self.fc4(x)

        return x
```


## 📂 Project Structure

```text
plant_leaf_detection/
│
├── backend/
│   ├── __pycache__/
│   ├── .env
│   ├── logger.py
│   ├── main.py
│   ├── model_schema.py
│   ├── model_train.ipynb
│   └── model_weights.pth
│
├── frontend/
│   └── templates/
│       └── index.html
│
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
└── template.py
```


# ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/MasudMallik/plant_leaf_detection.git

cd plant_leaf_detection
```

Create a virtual environment

### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv

source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# 🔑 Environment Variables

Create a `.env` file.

```env
GROQ_API_KEY=your_groq_api_key
```

---

# ▶️ Run the Application

```bash
uvicorn main:app --reload
```

Visit

```
http://127.0.0.1:8000
```


# 📦 Dependencies

Major libraries used in this project:

```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
FastAPI
PyTorch
TorchVision
Pillow
LangChain
langchain-groq
Jinja2
Uvicorn
python-multipart
```

Install all dependencies using

```bash
pip install -r requirements.txt
```

---

# 📈 Future Improvements

- 🌿 Support 100+ plant species
- 🍂 Plant disease detection
- 📱 Mobile application
- 📸 Camera capture support
- ☁️ Cloud deployment
- 📊 Prediction history
- 👤 User authentication
- 🌎 Multi-language support
- 🔍 Confidence visualization

---

# 🤝 Contributing

Contributions are always welcome.

```bash
Fork the repository

Create your feature branch

git checkout -b feature-name

Commit your changes

git commit -m "Added new feature"

Push to GitHub

git push origin feature-name

Open a Pull Request
```

---

# 👨‍💻 Author

## Masud Mallik

**AI & Machine Learning Undergraduate**

- GitHub: **https://github.com/MasudMallik**

---

# ⭐ Support

If you found this project helpful, please consider giving the repository a **⭐ Star**.

It motivates me to build more AI and Deep Learning projects!

---

# 📜 License

This project is licensed under the **MIT License**.

---

## 📬 Contact

Feel free to connect if you'd like to discuss AI, Deep Learning, FastAPI, or Python projects.

GitHub: **https://github.com/MasudMallik**