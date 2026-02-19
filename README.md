# Video Analysis Tool 🎥

An AI-powered video interview analysis system that provides comprehensive feedback on presentation skills, emotion detection, eye contact, and speech patterns.

## 🚀 Features

- **Emotion Detection**: Real-time facial emotion recognition using deep learning
- **Eye Contact Tracking**: Monitors and analyzes eye contact patterns throughout the video
- **Audio Analysis**: Transcribes speech and analyzes audio quality
- **Filler Word Detection**: Identifies and counts filler words (um, uh, like, etc.)
- **Interactive UI**: Built with Streamlit and Tkinter for easy use
- **AI-Powered Insights**: Integration with DeepSeek LangChain for intelligent analysis

## 📋 Requirements

```bash
pip install -r requirements.txt
```

Key Dependencies
OpenCV for video processing
TensorFlow/PyTorch for emotion detection
Streamlit for web interface
Speech recognition libraries
MediaPipe for facial landmarks
🎯 Usage
Streamlit Interface
bash
streamlit run StramlitUI.py
Tkinter Application
bash
python tkinter_app.py
Command Line
bash
python main.py --video path/to/video.mp4
📁 Project Structure
Code
Video-analysis/
├── StramlitUI.py              # Streamlit web interface
├── tkinter_app.py             # Desktop GUI application
├── main.py                    # Main processing script
├── emotion_detection/         # Emotion recognition module
├── eye_contact.py             # Eye contact tracking
├── eye_contact_optimized.py   # Optimized version
├── audio_analysis.py          # Audio processing
├── filler_word_summary.py     # Filler word detection
├── transcribe_audio.py        # Speech-to-text
├── lanchain_deepseek.py       # AI integration
└── requirements.txt           # Dependencies
🔧 Configuration
Create a .env file for API keys:

Code
DEEPSEEK_API_KEY=your_api_key_here
📊 Output
The tool provides:

Emotion distribution charts
Eye contact percentage and timeline
Filler word count and frequency
Audio transcription
Overall presentation score
AI-generated improvement suggestions
🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

📝 License
This project is open source and available under the MIT License.

Code

---

## 2. **Gatepass-backend**
**Description:** Node.js REST API backend for digital gate pass management system with PostgreSQL database, user authentication, and QR code generation

````markdown name=README.md
# Gatepass Backend 🚪

A robust Node.js backend API for managing digital gate passes with QR code generation, user authentication, and real-time tracking.

## 🚀 Features

- **User Authentication**: Secure JWT-based authentication system
- **Gate Pass Management**: Create, approve, and track gate passes
- **QR Code Generation**: Automatic QR code generation for passes
- **Role-Based Access Control**: Different permissions for students, guards, and admins
- **Database Migrations**: Knex.js for database version control
- **API Documentation**: Comprehensive API endpoints documentation

## 📋 Tech Stack

- **Runtime**: Node.js
- **Database**: PostgreSQL with Knex.js ORM
- **Authentication**: JWT tokens
- **QR Codes**: QR code generation library
- **Validation**: Express-validator

## 🛠️ Installation

```bash
# Install dependencies
npm install

# Setup database
npm run migrate:latest

# Seed database (optional)
npm run seed:run

# Start development server
npm run dev
```

## 📁 Project Structure

```
Gatepass-backend/
├── src/
│   ├── controllers/      # Request handlers
│   ├── models/          # Database models
│   ├── routes/          # API routes
│   ├── middleware/      # Auth & validation
│   └── utils/           # Helper functions
├── migrations/          # Database migrations
├── seeds/              # Database seeders
├── api-documentation.md # API docs
└── package.json
```

## 🔌 API Endpoints

See [api-documentation.md](api-documentation.md) for detailed endpoint information.

### Key Endpoints:
- `POST /api/auth/login` - User login
- `POST /api/auth/register` - User registration
- `GET /api/passes` - Get all passes
- `POST /api/passes` - Create new pass
- `PUT /api/passes/:id/approve` - Approve pass
- `GET /api/passes/:id/qr` - Get QR code

## ⚙️ Configuration

Create a `.env` file:
```env
DATABASE_URL=postgresql://user:password@localhost:5432/gatepass
JWT_SECRET=your_secret_key
PORT=3000
NODE_ENV=development
```

## 📊 Database Schema

- **users**: User accounts with roles
- **passes**: Gate pass records
- **approvals**: Approval workflow
- **logs**: Activity logging

## 🔒 Security

- Password hashing with bcrypt
- JWT token authentication
- Input validation and sanitization
- SQL injection prevention
- Rate limiting

## 🤝 Contributing

Contributions are welcome! Please follow the contribution guidelines.

## 📝 License

MIT License

