import React, { useState, useRef, useEffect } from 'react';
import { 
  Send, Upload, Shield, Activity, CircleAlert, Sparkles, 
  BookOpen, Image as ImageIcon, Database, BrainCircuit, RefreshCw, CheckCircle2 
} from 'lucide-react';
import { marked } from 'marked';

// Configure marked to render safe links
marked.setOptions({
  gfm: true,
  breaks: true
});

function App() {
  // Chat state
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      text: "Hello! I am your AI Medical Assistant. I can help you with:\n1. **Medical Scan Analysis** (Upload an X-ray, MRI, or skin photo below)\n2. **Reference Ingestion & Search** (Upload a medical PDF on the left, then ask questions about it)\n3. **General Medical Dialogue & Triage Guidance**\n\n*How can I assist you today?*",
      agent: 'Chat Agent',
      confidence: 100,
      rationale: 'Initial welcoming system state.'
    }
  ]);
  const [input, setInput] = useState('');
  const [selectedImage, setSelectedImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [sending, setSending] = useState(false);

  // Ingestion state
  const [ingestFile, setIngestFile] = useState(null);
  const [ingesting, setIngesting] = useState(false);
  const [ingestStatus, setIngestStatus] = useState('');

  // Agent flow tracking state
  const [activeStep, setActiveStep] = useState('IDLE'); // IDLE, ROUTING, VISION, RAG, CHAT, SYNTHESIS, COMPLETE
  const [lastRoute, setLastRoute] = useState(null);
  const [currentConfidence, setCurrentConfidence] = useState(null);
  const [currentRationale, setCurrentRationale] = useState('');
  const [citations, setCitations] = useState([]);

  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);

  // Auto-scroll chat to bottom on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, activeStep]);

  // Handle image upload selection
  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file && file.type.startsWith('image/')) {
      setSelectedImage(file);
      setImagePreview(URL.createObjectURL(file));
    }
  };

  // Clear selected image
  const clearImage = () => {
    setSelectedImage(null);
    setImagePreview(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  // Submit Query to API
  const handleSend = async (e) => {
    e.preventDefault();
    if (!input.trim() && !selectedImage) return;

    const userText = input;
    const userImg = selectedImage;
    const userImgPreview = imagePreview;

    // Add user message to UI
    setMessages(prev => [...prev, { 
      role: 'user', 
      text: userText, 
      imagePreview: userImgPreview 
    }]);

    // Reset inputs
    setInput('');
    clearImage();
    setSending(true);

    // Set routing visualization active
    setActiveStep('ROUTING');
    
    // Prepare multi-part form data
    const formData = new FormData();
    formData.append('query', userText || 'Analyze this medical scan.');
    
    // Package last 4 messages for conversational context
    const conversationHistory = messages
      .filter(m => m.role !== 'system')
      .slice(-4)
      .map(m => ({ role: m.role, text: m.text }));
    formData.append('history', JSON.stringify(conversationHistory));

    if (userImg) {
      formData.append('file', userImg);
    }

    try {
      // Simulate visual pipeline stages for better user understanding
      await new Promise(r => setTimeout(r, 600));

      const response = await fetch('http://localhost:8000/api/chat', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Server returned error code: ${response.status}`);
      }

      const data = await response.json();
      
      // Update pipeline stage to the active sub-agent
      setActiveStep(data.route); 
      setLastRoute(data.route);
      await new Promise(r => setTimeout(r, 1200)); // allow user to notice active node

      // Update pipeline stage to synthesizer
      setActiveStep('SYNTHESIS');
      await new Promise(r => setTimeout(r, 800));

      // Append assistant's answer
      setMessages(prev => [...prev, {
        role: 'assistant',
        text: data.final_answer,
        agent: data.agent,
        confidence: data.confidence,
        rationale: data.rationale
      }]);

      // Set side bar status info
      setCurrentConfidence(data.confidence);
      setCurrentRationale(data.rationale);
      setCitations(data.citations || []);
      setActiveStep('COMPLETE');

    } catch (error) {
      console.error("Error calling assistant API:", error);
      setMessages(prev => [...prev, {
        role: 'assistant',
        text: `**Connection Error**: Failed to reach backend API. Make sure your FastAPI server is running on port 8000.\n\n*Error details: ${error.message}*`,
        agent: 'System Guard',
        confidence: 0,
        rationale: 'Network routing error encountered.'
      }]);
      setActiveStep('IDLE');
    } finally {
      setSending(false);
    }
  };

  // Submit Reference Document (PDF)
  const handleIngest = async (e) => {
    e.preventDefault();
    if (!ingestFile) return;

    setIngesting(true);
    setIngestStatus('Parsing PDF with Docling...');

    const formData = new FormData();
    formData.append('file', ingestFile);

    try {
      const response = await fetch('http://localhost:8000/api/ingest', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      if (response.ok && data.success) {
        setIngestStatus('Successfully ingested and embedded document into Qdrant!');
        setIngestFile(null);
        // Clear input file
        document.getElementById('pdf-file-input').value = '';
      } else {
        setIngestStatus(`Ingestion Error: ${data.detail || 'Failed to parse document'}`);
      }
    } catch (error) {
      console.error("Ingestion failed:", error);
      setIngestStatus(`Ingestion Failed: ${error.message}`);
    } finally {
      setIngesting(false);
    }
  };

  return (
    <div className="app-container">
      {/* Top Navigation */}
      <header className="glass-panel app-header">
        <div className="header-logo-container">
          <div className="logo-icon-bg">
            <BrainCircuit className="w-6 h-6" />
          </div>
          <div className="header-title-text">
            <h1>MedAssist AI</h1>
            <p>Multi-Agent Diagnostics Console</p>
          </div>
        </div>

        <div className="header-status-container">
          <div className="status-badge gemini">
            <span className="status-indicator-dot pulse"></span>
            Ollama Offline Stack
          </div>
          <div className="status-badge qdrant">
            <span className="status-indicator-dot pulse"></span>
            Qdrant DB Connected
          </div>
        </div>
      </header>

      {/* Main Panel Grid */}
      <div className="dashboard-grid">
        
        {/* Panel 1: Document Ingest & File upload controls (Left Sidebar) */}
        <div className="left-sidebar">
          {/* Doc Ingestion Module */}
          <div className="glass-panel panel-card">
            <div className="panel-header indigo">
              <Database className="w-5 h-5" />
              <span>Reference Literature Ingest</span>
            </div>
            <p className="panel-description">
              Upload PDF clinical guidelines, drug brochures, or textbooks. Docling will parse layouts/tables and store them in Qdrant.
            </p>

            <form onSubmit={handleIngest} className="ingest-form">
              <div className="drop-zone">
                <input 
                  type="file" 
                  id="pdf-file-input"
                  accept=".pdf"
                  onChange={(e) => setIngestFile(e.target.files[0])}
                />
                <BookOpen className="drop-zone-icon" />
                <span className="drop-zone-text">
                  {ingestFile ? ingestFile.name : 'Select clinical guidelines PDF'}
                </span>
                <span className="drop-zone-subtext">PDF max 20MB</span>
              </div>

              <button
                type="submit"
                disabled={!ingestFile || ingesting}
                className="btn-primary"
              >
                {ingesting ? (
                  <>
                    <RefreshCw className="w-4 h-4 animate-spin" />
                    <span>Processing...</span>
                  </>
                ) : (
                  <>
                    <Upload className="w-4 h-4" />
                    <span>Load into RAG Database</span>
                  </>
                )}
              </button>
            </form>

            {ingestStatus && (
              <div className={`ingest-status-msg ${ingestStatus.startsWith('Error') || ingestStatus.startsWith('Failed') ? 'error' : 'info'}`}>
                {ingestStatus}
              </div>
            )}
          </div>

          {/* Supported Diagnostics Box */}
          <div className="glass-panel panel-card" style={{ flex: 1 }}>
            <div className="panel-header teal">
              <Activity className="w-5 h-5" />
              <span>Diagnostic Capability</span>
            </div>
            <p className="panel-description">
              Attach an image in chat to activate local Moondream Vision analysis:
            </p>
            <div className="capability-grid">
              <div className="capability-item">
                <span className="capability-item-title">Chest X-Ray</span>
                <span className="capability-item-desc">Pneumonia, Lung</span>
              </div>
              <div className="capability-item">
                <span className="capability-item-title">Brain MRI</span>
                <span className="capability-item-desc">Structural, Tumor</span>
              </div>
              <div className="capability-item">
                <span className="capability-item-title">Skin Lesions</span>
                <span className="capability-item-desc">Dermatology, Moles</span>
              </div>
              <div className="capability-item">
                <span className="capability-item-title">Retinal Fundus</span>
                <span className="capability-item-desc">Vessels, Macula</span>
              </div>
            </div>
          </div>
        </div>

        {/* Panel 2: Chat Console & Flow Visualization (Center Column) */}
        <div className="center-column">
          
          {/* Agent Flow Visualizer */}
          <div className="glass-panel flow-visualizer-container">
            <div className="flow-title">
              Live Agent Execution Pipeline
            </div>
            <div className="flow-nodes-wrapper">
              
              {/* User Input node */}
              <div className="flow-node-item">
                <div className={`flow-node-icon-box ${activeStep !== 'IDLE' && activeStep !== 'COMPLETE' ? 'active-routing' : ''}`}>
                  <Send className="w-5 h-5" />
                </div>
                <span>Input</span>
              </div>

              <div className="flow-connector-line"></div>

              {/* Router node */}
              <div className="flow-node-item">
                <div className={`flow-node-icon-box ${activeStep === 'ROUTING' ? 'active-routing' : ''}`}>
                  <BrainCircuit className="w-5 h-5" />
                </div>
                <span>Router</span>
              </div>

              <div className="flow-connector-line"></div>

              {/* Sub-agents fork container */}
              <div className="flow-specialists-group">
                {/* Vision Agent */}
                <div className="specialist-node">
                  <div className={`specialist-icon-box ${activeStep === 'VISION' || (activeStep === 'COMPLETE' && lastRoute === 'VISION') ? 'active-vision' : ''}`}>
                    <ImageIcon className="w-4 h-4" />
                  </div>
                  <span>Vision</span>
                </div>

                {/* RAG Agent */}
                <div className="specialist-node">
                  <div className={`specialist-icon-box ${activeStep === 'RAG' || (activeStep === 'COMPLETE' && lastRoute === 'RAG') ? 'active-rag' : ''}`}>
                    <Database className="w-4 h-4" />
                  </div>
                  <span>RAG</span>
                </div>

                {/* Chat Agent */}
                <div className="specialist-node">
                  <div className={`specialist-icon-box ${activeStep === 'CHAT' || (activeStep === 'COMPLETE' && lastRoute === 'CHAT') ? 'active-chat' : ''}`}>
                    <Sparkles className="w-4 h-4" />
                  </div>
                  <span>Chat</span>
                </div>
              </div>

              <div className="flow-connector-line"></div>

              {/* Synthesizer node */}
              <div className="flow-node-item">
                <div className={`flow-node-icon-box ${activeStep === 'SYNTHESIS' ? 'active-synthesis' : ''}`}>
                  <Shield className="w-5 h-5" />
                </div>
                <span>Synthesizer</span>
              </div>

              <div className="flow-connector-line"></div>

              {/* Output node */}
              <div className="flow-node-item">
                <div className={`flow-node-icon-box ${activeStep === 'COMPLETE' ? 'active-complete' : ''}`}>
                  <CheckCircle2 className="w-5 h-5" />
                </div>
                <span>Answer</span>
              </div>

            </div>
          </div>

          {/* Chat Interface Container */}
          <div className="glass-panel chat-container">
            {/* Chat Messages */}
            <div className="chat-messages-area">
              {messages.map((msg, index) => (
                <div 
                  key={index} 
                  className={`chat-message-bubble ${msg.role === 'user' ? 'user' : 'assistant'}`}
                >
                  {/* Message Source Header */}
                  {msg.role === 'assistant' && (
                    <div className="message-agent-header">
                      <BrainCircuit className="w-3.5 h-3.5" />
                      <span>{msg.agent}</span>
                      {msg.confidence !== undefined && (
                        <span className={`message-agent-confidence-pill ${msg.confidence >= 90 ? 'high' : msg.confidence >= 70 ? 'medium' : 'low'}`}>
                          {msg.confidence}% Confidence
                        </span>
                      )}
                    </div>
                  )}

                  {/* Render uploaded image inside chat */}
                  {msg.imagePreview && (
                    <img 
                      src={msg.imagePreview} 
                      alt="User uploaded scan" 
                      className="msg-scan-attachment"
                    />
                  )}

                  {/* Message Text Content */}
                  <div 
                    className="text-sm markdown-content"
                    dangerouslySetInnerHTML={{ __html: marked(msg.text) }}
                  />
                </div>
              ))}

              {/* Active Pipeline Status Message */}
              {sending && (
                <div className="chat-processing-indicator">
                  <RefreshCw className="spin-loader" />
                  <span>
                    {activeStep === 'ROUTING' && 'Router Agent mapping intent...'}
                    {activeStep === 'VISION' && 'Vision Agent analyzing scan findings...'}
                    {activeStep === 'RAG' && 'RAG Agent searching semantic database...'}
                    {activeStep === 'CHAT' && 'Chat Agent generating medical advice...'}
                    {activeStep === 'SYNTHESIS' && 'Synthesizer Agent applying safety checks...'}
                  </span>
                </div>
              )}
              
              <div ref={messagesEndRef} />
            </div>

            {/* Chat Input & Attacher */}
            <form onSubmit={handleSend} className="chat-input-bar-form">
              {imagePreview && (
                <div className="chat-input-image-preview-badge">
                  <img src={imagePreview} alt="Preview" />
                  <span>Selected Scan Attached</span>
                  <button 
                    type="button" 
                    onClick={clearImage}
                  >
                    Remove
                  </button>
                </div>
              )}

              <div className="chat-input-controls-row">
                {/* Image Attach trigger */}
                <button
                  type="button"
                  onClick={() => fileInputRef.current?.click()}
                  className="btn-attach-scan"
                  title="Attach Scan Image"
                >
                  <ImageIcon className="w-5 h-5" />
                </button>
                <input 
                  type="file" 
                  ref={fileInputRef}
                  accept="image/*"
                  className="hidden"
                  onChange={handleImageChange}
                />

                <input
                  type="text"
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  placeholder="Ask a medical query or consult clinical scans..."
                  disabled={sending}
                  className="chat-text-input"
                />

                <button
                  type="submit"
                  disabled={sending || (!input.trim() && !selectedImage)}
                  className="btn-send-message"
                >
                  <Send className="w-5 h-5" />
                </button>
              </div>
            </form>
          </div>
        </div>

        {/* Panel 3: Metrics, Guardrails, Citations (Right Sidebar) */}
        <div className="right-sidebar">
          
          {/* Confidence and Rationale box */}
          <div className="glass-panel panel-card">
            <div className="panel-header amber">
              <Shield className="w-5 h-5" />
              <span>Safety & Confidence</span>
            </div>

            <div className="metrics-panel-container">
              <div className="gauge-svg-wrapper">
                {/* Visual Gauge */}
                <svg className="w-32 h-32 transform -rotate-90" style={{ transform: 'rotate(-90deg)' }}>
                  <circle
                    cx="64"
                    cy="64"
                    r="52"
                    stroke="rgba(255,255,255,0.03)"
                    strokeWidth="8"
                    fill="transparent"
                  />
                  <circle
                    cx="64"
                    cy="64"
                    r="52"
                    stroke={currentConfidence >= 85 ? "#10b981" : currentConfidence >= 65 ? "#f59e0b" : "#f43f5e"}
                    strokeWidth="8"
                    fill="transparent"
                    strokeDasharray={2 * Math.PI * 52}
                    strokeDashoffset={2 * Math.PI * 52 * (1 - (currentConfidence || 0) / 100)}
                    style={{ transition: 'stroke-dashoffset 0.8s ease' }}
                  />
                </svg>
                <div className="gauge-value-text-box">
                  <span className="gauge-score-number">
                    {currentConfidence !== null ? `${currentConfidence}%` : 'N/A'}
                  </span>
                  <span className="gauge-score-label">Confidence</span>
                </div>
              </div>
            </div>

            <div className="rationale-quote-box">
              <span className="rationale-quote-box-label">Clinical Rationale</span>
              <p>
                {currentRationale || 'No diagnostic rationale generated yet. Execute a query to check confidence.'}
              </p>
            </div>

            <div className="active-guardrail-status-pill">
              <span>Guardrail Status:</span>
              <span>
                <CheckCircle2 className="w-4 h-4" />
                ACTIVE SAFE
              </span>
            </div>
          </div>

          {/* RAG Citations Panel */}
          <div className="glass-panel panel-card" style={{ flex: 1 }}>
            <div className="panel-header indigo">
              <BookOpen className="w-5 h-5" />
              <span>Reference Citations</span>
            </div>
            <p className="panel-description">
              Clinical citations retrieved from your Qdrant vector database:
            </p>

            <div className="citations-list-container">
              {citations.length > 0 ? (
                citations.map((cite, index) => (
                  <div key={index} className="citation-card-item">
                    <div className="citation-card-header">
                      <span>{cite.source_file}</span>
                      <span>
                        Match: {Math.round(cite.score * 100)}%
                      </span>
                    </div>
                    <span className="citation-card-sublabel">Chunk Reference #{index + 1}</span>
                  </div>
                ))
              ) : (
                <div className="empty-citations-box">
                  <Database className="empty-citations-icon" />
                  <span>No citations active.</span>
                  <p>RAG answers will automatically generate document references.</p>
                </div>
              )}
            </div>
          </div>
        </div>

      </div>
    </div>
  );
}

export default App;
