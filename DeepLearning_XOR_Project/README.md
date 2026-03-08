# 🧠 Çok Katmanlı Perceptron ile XOR Problemi Çözümü

Bu çalışma, yapay sinir ağlarının tarihsel gelişiminde önemli bir yere sahip olan **XOR (Exclusive OR)** probleminin, çok katmanlı sinir ağları kullanılarak nasıl çözülebileceğini uygulamalı olarak göstermektedir. XOR problemi, doğrusal olarak ayrılamayan veri yapılarına klasik perceptron modellerinin neden başarısız olduğunu açıklayan temel örneklerden biridir.

---

## 📝 Problem Tanımı

XOR problemi, iki girişli bir mantık kapısıdır ve çıktısı aşağıdaki gibidir:

| x₁ | x₂ | Çıkış |
|----|----|------|
| 0  | 0  | 0 |
| 0  | 1  | 1 |
| 1  | 0  | 1 |
| 1  | 1  | 0 |

Bu veri yapısı **lineer olarak ayrılamaz**. Yani veriler tek bir doğru ile iki sınıfa ayrıştırılamaz. Bu nedenle **tek katmanlı perceptronlar XOR problemini çözemez**. Ancak gizli katman içeren **çok katmanlı perceptron (MLP)** mimarileri doğrusal olmayan karar sınırları öğrenerek bu problemi başarıyla çözebilir.

---

## ⚙️ Kullanılan Teknolojiler

Bu çalışmada aşağıdaki teknolojiler kullanılmıştır:

- 🐍 **Python 3**
- 🔥 **PyTorch** (Derin öğrenme modeli oluşturmak için)
- 📊 **NumPy** (Veri işlemleri için)
- 📈 **Matplotlib** (Grafik ve görselleştirme için)

---

## 🏗️ Model Mimarisi

Model, PyTorch kullanılarak **Çok Katmanlı Perceptron (MLP)** mimarisi ile oluşturulmuştur.

Model yapısı:

- **Giriş Katmanı:** 2 nöron *(x₁, x₂)*
- **Gizli Katman:** 8 nöron ve **Sigmoid aktivasyon fonksiyonu**
- **Çıkış Katmanı:** 1 nöron ve **Sigmoid aktivasyon fonksiyonu**

Eğitim sürecinde kullanılan parametreler:

- **Kayıp Fonksiyonu:** Mean Squared Error (MSE)
- **Optimizasyon Algoritması:** Adam Optimizer
- **Öğrenme Oranı:** 0.01
- **Epoch Sayısı:** 5000

---

## 📈 Model Performansı

Model eğitim sürecinde kayıp değerini önemli ölçüde azaltarak XOR ilişkisini başarıyla öğrenmiştir. Eğitim sırasında elde edilen **kayıp grafiği** ve modelin öğrendiği **karar sınırı (decision boundary)** aşağıda gösterilmektedir.

### Eğitim Kayıp Grafiği ve Karar Sınırı

![XOR Sonuç Grafiği](./results.png)

Grafikte görüldüğü gibi model, doğrusal olmayan bir karar sınırı oluşturarak XOR veri noktalarını doğru şekilde sınıflandırmayı öğrenmiştir.

---

## 🚀 Projeyi Çalıştırma

Projeyi yerel ortamınızda çalıştırmak için aşağıdaki adımları izleyebilirsiniz.

### 1️⃣ Gerekli kütüphaneleri yükleyin

```bash
pip install torch numpy matplotlib
```

### 2️⃣ Modeli çalıştırın

```bash
python xor_model.py
```

Model eğitilecek ve eğitim sürecine ait grafikler ekranda gösterilecektir.

---

## 📌 Sonuç

Bu deney, tek katmanlı perceptronların doğrusal olmayan problemleri çözemediğini ve gizli katman içeren çok katmanlı sinir ağlarının bu tür problemleri başarıyla öğrenebildiğini göstermektedir. XOR problemi, derin öğrenme modellerinin doğrusal olmayan veri yapılarından anlamlı ilişkiler öğrenebilmesini açıklayan klasik ve önemli bir örnektir.
