# Road Traffic Accident Prediction System for India 🚗

A machine learning-based web application that predicts road accident severity using real-time Indian data sources including weather conditions, road types, traffic density, and geographic information.

## 🌟 Features

- **Real-time Accident Risk Prediction**: Get instant risk assessment based on current conditions
- **Indian Data Integration**: Uses data patterns from Ministry of Road Transport & Highways (MoRTH)
- **Weather Data**: Integrates with India Meteorological Department (IMD) APIs
- **Interactive Dashboard**: Comprehensive data visualization and analytics
- **State-wise Analysis**: Regional accident pattern analysis for all Indian states
- **Multiple ML Models**: Random Forest, XGBoost, and other algorithms for accurate predictions

## 🚀 Live Demo

Visit the deployed application: [Your-App-URL-Here]

## 📊 Screenshots

[Add screenshots of your application here]

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **Machine Learning**: Scikit-learn, Random Forest
- **Data Visualization**: Plotly
- **APIs**: IMD Weather API, Open Government Data
- **Deployment**: Streamlit Community Cloud

## 📈 Model Performance

- **Algorithm**: Random Forest Classifier
- **Accuracy**: ~77%
- **Features**: 11 input parameters
- **Training Data**: 3,000+ accident records

## 🗂️ Project Structure

```
road-accident-prediction/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── data/                 # Data files (optional)
└── models/               # Trained models (optional)
```

## 🔧 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- Git

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/road-accident-prediction.git
   cd road-accident-prediction
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser** and go to `http://localhost:8501`

## 🚀 Deployment Guide

### Deploy to Streamlit Community Cloud (FREE)

1. **Create GitHub Repository**
   - Push your code to GitHub
   - Make sure your repository is public

2. **Deploy to Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Set main file path to `app.py`
   - Click "Deploy"

3. **Your app will be live** at `https://your-app-name.streamlit.app`

### Alternative Deployment Options

- **Heroku**: See deployment guide in docs
- **AWS EC2**: For production deployments
- **Google Cloud**: Using Cloud Run
- **Azure**: Using Container Apps

## 📝 Usage

### Dashboard
- View comprehensive accident statistics
- Analyze trends by state, time, and weather
- Interactive charts and visualizations

### Accident Prediction
- Input current conditions (location, time, weather)
- Get real-time risk assessment
- View probability distributions
- Identify risk factors

### Data Analysis
- Explore correlation between different factors
- Time-based analysis (hourly, daily, monthly)
- Weather impact on accident severity

## 🔗 Data Sources

1. **Ministry of Road Transport & Highways (MoRTH)**
   - Historical accident data
   - Road infrastructure information

2. **India Meteorological Department (IMD)**
   - Real-time weather data
   - Historical weather patterns

3. **Open Government Data Platform India**
   - District-wise statistics
   - Road safety data

## 🎯 Key Insights

- **Night hours (22:00-05:00)** show higher accident severity
- **Rainy and foggy weather** increases accident risk by 20-30%
- **National Highways** have higher fatal accident rates
- **Two-wheelers** are most vulnerable vehicle type
- **High traffic density** correlates with accident frequency

## 🔮 Future Enhancements

- [ ] Real-time traffic data integration
- [ ] GPS-based location detection
- [ ] Mobile app development
- [ ] Advanced deep learning models
- [ ] IoT sensor data integration
- [ ] Emergency response optimization

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This application is for educational and research purposes. The synthetic data is based on real patterns but should not be used for actual emergency response or critical decision making without proper validation with real-time data sources.

## 📞 Support

For questions or support:
- 📧 Email: contact@roadaccidentprediction.in
- 💬 Issues: [GitHub Issues](https://github.com/yourusername/road-accident-prediction/issues)
- 📖 Documentation: [Wiki](https://github.com/yourusername/road-accident-prediction/wiki)

## 🙏 Acknowledgments

- Ministry of Road Transport & Highways, Government of India
- India Meteorological Department
- Open Government Data Platform India
- Streamlit Community
- Scikit-learn and Plotly developers

---

⭐ **Star this repository if you found it helpful!**
