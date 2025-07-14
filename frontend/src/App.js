import React, { useState, useEffect } from 'react';
import {
  Box,
  Grid,
  Paper,
  Typography,
  TextField,
  Button,
  Divider,
  Slider,
  CircularProgress,
  IconButton,
  Collapse,
  Tooltip
} from '@mui/material';
import { styled, keyframes } from '@mui/material/styles';
import DeleteIcon from '@mui/icons-material/Delete';
import EditIcon from '@mui/icons-material/Edit';
import CheckIcon from '@mui/icons-material/Check';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ExpandLessIcon from '@mui/icons-material/ExpandLess';
import AddIcon from '@mui/icons-material/Add';

//
// Animation for chat bubbles.
//
const fadeIn = keyframes`
  from { opacity: 0; transform: translateY(4px); }
  to { opacity: 1; transform: translateY(0); }
`;

//
// ChatBubble styled component.
//
const ChatBubble = styled(Paper)(({ theme, sender }) => ({
  padding: theme.spacing(1.5, 2),
  backgroundColor: sender === 'bot' ? '#f5f5f5' : '#dcf8c6',
  border: sender === 'bot' ? '1px solid #ddd' : 'none',
  color: '#333',
  borderRadius: 16,
  maxWidth: '75%',
  animation: `${fadeIn} 0.3s ease-in-out`,
}));

//
// Define conversation steps.
//
const STEPS = {
  BEAT: 0,
  BEAT_CONFIRM: 1,
  CHARACTER_NAME: 2,
  CHARACTER_DESCRIPTION: 3,
  SETTING: 4,
  GENRE: 5,
  STYLE: 6,
  CONFIRM: 7,
  GENERATE: 8,
  EXTEND: 9,
};

//
// Predefined colors for characters.
//
const characterColors = ["#f44336", "#4caf50", "#2196f3", "#ff9800", "#9c27b0"];

// cover-art style options (keep the empty key for “skip”)
const COVER_OPTIONS = [
  { key: '', label: 'No image (skip)' },

  // ── Traditional media ──
  { key: 'oil_paint',         label: 'Oil paint' },
  { key: 'acrylic',           label: 'Acrylic' },
  { key: 'watercolour',       label: 'Watercolour' },
  { key: 'gouache',           label: 'Gouache' },
  { key: 'ink_wash',          label: 'Ink wash' },
  { key: 'pencil_sketch',     label: 'Pencil sketch' },
  { key: 'pastel',            label: 'Pastel' },
  { key: 'charcoal',          label: 'Charcoal' },

  // ── Print & graphic ──
  { key: 'linocut',           label: 'Linocut print' },
  { key: 'woodblock',         label: 'Woodblock print' },
  { key: 'silkscreen',        label: 'Silkscreen / Pop-art' },
  { key: 'vector_flat',       label: 'Vector flat' },
  { key: 'papercut',          label: 'Paper-cut' },

  // ── Historical movements ──
  { key: 'renaissance',       label: 'Renaissance' },
  { key: 'baroque',           label: 'Baroque' },
  { key: 'impressionist',     label: 'Impressionist' },
  { key: 'post_impressionist',label: 'Post-Impressionist' },
  { key: 'art_nouveau',       label: 'Art Nouveau' },
  { key: 'art_deco',          label: 'Art Deco' },
  { key: 'cubist',            label: 'Cubist' },
  { key: 'surrealist',        label: 'Surrealist' },
  { key: 'expressionist',     label: 'Expressionist' },
  { key: 'futurist',          label: 'Futurist' },

  // ── Modern / pop culture ──
  { key: 'comic_retro',       label: 'Retro comic' },
  { key: 'manga_bw',          label: 'Manga (B/W)' },
  { key: 'anime',             label: 'Anime' },
  { key: 'pixel_art',         label: 'Pixel art' },
  { key: 'low_poly',          label: 'Low-poly' },
  { key: 'vaporwave',         label: 'Vaporwave' },
  { key: 'steampunk',         label: 'Steampunk' },
  { key: 'cyberpunk',         label: 'Cyberpunk' },
  { key: 'dieselpunk',        label: 'Dieselpunk' },
  { key: 'dark_fantasy',      label: 'Dark fantasy' },
  { key: 'neon_noir',         label: 'Neon noir' },

  // ── Photographic / cinematic ──
  { key: 'photoreal_cinema',  label: 'Photoreal cinema' },
  { key: 'matte_painting',    label: 'Matte painting' },
  { key: 'isometric',         label: 'Isometric' },
  { key: 'concept_art',       label: 'Concept art' },

  // ── Decorative & misc ──
  { key: 'folk_art',          label: 'Folk art' },
  { key: 'silhouette',        label: 'Silhouette' },
  { key: 'chalkboard',        label: 'Chalkboard' },
  { key: 'embroidery',        label: 'Embroidery' },
  { key: 'stained_glass',     label: 'Stained glass' },
];


export default function App() {
  // Conversation state.
  const [step, setStep] = useState(STEPS.BEAT);
  const [data, setData] = useState({
    beats: [],
    characters: [], // Each character: { name, description, color }
    setting: '',
    genre: '',
    style: '',
  });
  // Temporary state for current character name.
  const [currentCharName, setCurrentCharName] = useState("");

  // Editing states for parameters.
  const [editingSetting, setEditingSetting] = useState(false);
  const [editingGenre, setEditingGenre] = useState(false);
  const [editingStyle, setEditingStyle] = useState(false);
  const [tempSetting, setTempSetting] = useState("");
  const [tempGenre, setTempGenre] = useState("");
  const [tempStyle, setTempStyle] = useState("");

  // Toggle collapsible sections for parameters.
  const [showBeats, setShowBeats] = useState(true);
  const [showCharacters, setShowCharacters] = useState(true);
  const [showOtherParams, setShowOtherParams] = useState(true);

  // Additional controls for adding beats/characters from the parameters panel.
  const [newBeat, setNewBeat] = useState("");
  const [showNewBeatInput, setShowNewBeatInput] = useState(false);
  const [newCharacterName, setNewCharacterName] = useState("");
  const [newCharacterDesc, setNewCharacterDesc] = useState("");
  const [showNewCharacterInput, setShowNewCharacterInput] = useState(false);

  // Initial welcome message with extra backend details.
  const initialMessage = `Welcome! I'm ProseGenerator – an AI-powered creative writing assistant built with React, Material‑UI, FastAPI, OpenAI GPT‑4, and Pinecone.
I use advanced NLP, Retrieval-Augmented Generation (RAG) techniques, and a pre-built vector database (Pinecone) to transform your story beats into a narrative.
(For your information, data ingestion (embedding BookSum) has already been completed and stored in Pinecone.)
Let's begin! Please enter your first Story Beat.`;

  // Chat messages.
  const [messages, setMessages] = useState([{ sender: 'bot', text: initialMessage }]);
  const [currentPrompt, setCurrentPrompt] = useState("Type your message here...");
  const [userInput, setUserInput] = useState("");
  const [generatedStory, setGeneratedStory] = useState("");

  // Generation control parameters.
  const [temperature, setTemperature] = useState(0.7);
  const [desiredLength, setDesiredLength] = useState(1000);
  // Cover-art
  const [coverStyle, setCoverStyle] = useState('');   // key sent to backend
  const [coverUrl,   setCoverUrl]   = useState(null); // URL returned by backend

  // NEW
  const [visualPrompt,  setVisualPrompt]  = useState('');   // free-form text


  /*
    Layout modes:
      - BEFORE generation:
          "both"     → Left panel occupies 9 columns and right panel (parameters) occupies 3 columns.
          "chatOnly" → Only the left (chat) panel is visible.
      - AFTER generation:
          "both"      → Left (chat) occupies 8 columns and right (generated story) occupies 4 columns.
          "storyOnly" → Only the right panel is visible.
  */
  const [layoutMode, setLayoutMode] = useState("both");
  const [loading, setLoading] = useState(false);

  // Debug logging.
  useEffect(() => {
    console.log("Step:", step);
    console.log("Data:", data);
    console.log("Messages:", messages);
    console.log("Generated story exists?", generatedStory !== "");
    console.log("Layout mode:", layoutMode);
  }, [step, data, messages, generatedStory, layoutMode]);

  // Remove a character by index.
  const removeCharacter = (index) => {
    setData(prevData => ({
      ...prevData,
      characters: prevData.characters.filter((_, i) => i !== index)
    }));
  };

  // Remove a beat by index.
  const removeBeat = (index) => {
    setData(prevData => ({
      ...prevData,
      beats: prevData.beats.filter((_, i) => i !== index)
    }));
  };

  // Toggle edit mode for a parameter.
  const toggleEdit = (param) => {
    if (param === "setting") {
      setTempSetting(data.setting);
      setEditingSetting((prev) => !prev);
    } else if (param === "genre") {
      setTempGenre(data.genre);
      setEditingGenre((prev) => !prev);
    } else if (param === "style") {
      setTempStyle(data.style);
      setEditingStyle((prev) => !prev);
    }
  };

  // Save edited parameter.
  const saveEdit = (param) => {
    if (param === "setting") {
      setData(prev => ({ ...prev, setting: tempSetting }));
      setEditingSetting(false);
    } else if (param === "genre") {
      setData(prev => ({ ...prev, genre: tempGenre }));
      setEditingGenre(false);
    } else if (param === "style") {
      setData(prev => ({ ...prev, style: tempStyle }));
      setEditingStyle(false);
    }
  };

  // A simple function to highlight character names in the generated story.
  const highlightStory = (text) => {
    let highlighted = text;
    data.characters.forEach((char) => {
      const regex = new RegExp(`\\b(${char.name})\\b`, "gi");
      highlighted = highlighted.replace(
        regex,
        `<span style="color: ${char.color}; font-weight: bold;">$1</span>`
      );
    });
    return highlighted;
  };

  // Reset conversation.
  const startOver = () => {
    console.log("startOver() called");
    setStep(STEPS.BEAT);
    setData({ beats: [], characters: [], setting: '', genre: '', style: '' });
    setCurrentCharName("");
    setUserInput("");
    setGeneratedStory("");
    setTemperature(0.7);
    setDesiredLength(1000);
    setLayoutMode("both");
    setMessages([{ sender: 'bot', text: initialMessage }]);
    setCurrentPrompt("Type your message here...");
    console.log("After startOver(), step:", STEPS.BEAT);
  };

  // Handler for sending user input in the conversation.
  const handleSend = async (e) => {
    e.preventDefault();
    const input = userInput.trim();
    if (!input) return;
    console.log("User input:", input);

    // Append user's message.
    setMessages(prev => [...prev, { sender: 'user', text: input }]);

    // Allow "start over".
    if (input.toLowerCase() === "start over") {
      startOver();
      return;
    }

    // Conversation flow.
    if (step === STEPS.BEAT) {
      setData(prev => ({ ...prev, beats: [...prev.beats, input] }));
      setMessages(prev => [...prev, { sender: 'bot', text: "Beat added. Do you want to add another beat? (yes/no)" }]);
      setCurrentPrompt("Do you want to add another beat? (yes/no)");
      setStep(STEPS.BEAT_CONFIRM);
    } else if (step === STEPS.BEAT_CONFIRM) {
      if (input.toLowerCase() === 'yes') {
        setMessages(prev => [...prev, { sender: 'bot', text: "Please enter your next Story Beat:" }]);
        setCurrentPrompt("Enter your next Story Beat:");
        setStep(STEPS.BEAT);
      } else {
        setMessages(prev => [...prev, { sender: 'bot', text: "Please enter the name of your first character:" }]);
        setCurrentPrompt("Enter character name:");
        setStep(STEPS.CHARACTER_NAME);
      }
    } else if (step === STEPS.CHARACTER_NAME) {
      setCurrentCharName(input);
      setMessages(prev => [...prev, { sender: 'bot', text: "Now enter the character's description:" }]);
      setCurrentPrompt("Enter character description:");
      setStep(STEPS.CHARACTER_DESCRIPTION);
    } else if (step === STEPS.CHARACTER_DESCRIPTION) {
      setData(prev => {
        const newChar = {
          name: currentCharName,
          description: input,
          color: characterColors[prev.characters.length % characterColors.length]
        };
        return { ...prev, characters: [...prev.characters, newChar] };
      });
      setMessages(prev => [...prev, { sender: 'bot', text: "Character saved. Do you want to add another character? (yes/no)" }]);
      setCurrentPrompt("Add another character? (yes/no)");
      setStep(STEPS.CHARACTERS);
    } else if (step === STEPS.CHARACTERS) {
      if (input.toLowerCase() === 'yes') {
        setMessages(prev => [...prev, { sender: 'bot', text: "Please enter the name of your next character:" }]);
        setCurrentPrompt("Enter character name:");
        setStep(STEPS.CHARACTER_NAME);
      } else {
        setMessages(prev => [...prev, { sender: 'bot', text: "Characters saved. Now, please enter the Setting of your story:" }]);
        setCurrentPrompt("Enter the Setting:");
        setStep(STEPS.SETTING);
      }
    } else if (step === STEPS.SETTING) {
      setData(prev => ({ ...prev, setting: input }));
      setMessages(prev => [...prev, { sender: 'bot', text: "Setting saved. Now, please enter the Genre:" }]);
      setCurrentPrompt("Enter the Genre:");
      setStep(STEPS.GENRE);
    } else if (step === STEPS.GENRE) {
      setData(prev => ({ ...prev, genre: input }));
      setMessages(prev => [...prev, { sender: 'bot', text: "Genre saved. Now, please enter the desired Style:" }]);
      setCurrentPrompt("Enter the Style:");
      setStep(STEPS.STYLE);
    } else if (step === STEPS.STYLE) {
      setData(prev => ({ ...prev, style: input }));
      // Construct final summary message as plain text.
      const summaryText = `
All parameters saved.
Beats:
${data.beats.map((beat, idx) => `${idx + 1}. ${beat}`).join("\n")}
Characters: ${data.characters.map(c => c.name).join(", ")}
Setting: ${data.setting}
Genre: ${data.genre}
Style: ${data.style}

Generate your story? (yes/no)
      `;
      setMessages(prev => [...prev, { sender: 'bot', text: summaryText }]);
      setCurrentPrompt("Generate your story? (yes/no)");
      setStep(STEPS.CONFIRM);
    } else if (step === STEPS.CONFIRM) {
      if (input.toLowerCase() === 'yes') {
        setMessages(prev => [...prev, { sender: 'bot', text: "Generating your story... Please wait." }]);
        setCurrentPrompt("Generating...");
        setStep(STEPS.GENERATE);
        await generateStory();
      } else {
        setMessages(prev => [...prev, { sender: 'bot', text: "Story generation cancelled. To try again, type 'start over'." }]);
        setCurrentPrompt("");
      }
    }
    setUserInput("");
  };

  // Function to add a new beat from the parameters panel.
  const handleAddBeat = () => {
    if (newBeat.trim() !== "") {
      setData(prev => ({ ...prev, beats: [...prev.beats, newBeat.trim()] }));
      setNewBeat("");
      setShowNewBeatInput(false);
    }
  };

  // Function to cancel new beat input.
  const cancelNewBeat = () => {
    setNewBeat("");
    setShowNewBeatInput(false);
  };

  // Function to add a new character from the parameters panel.
  const handleAddCharacter = () => {
    if (newCharacterName.trim() !== "" && newCharacterDesc.trim() !== "") {
      const newChar = {
        name: newCharacterName.trim(),
        description: newCharacterDesc.trim(),
        color: characterColors[data.characters.length % characterColors.length]
      };
      setData(prev => ({ ...prev, characters: [...prev.characters, newChar] }));
      setNewCharacterName("");
      setNewCharacterDesc("");
      setShowNewCharacterInput(false);
    }
  };

  // Function to cancel new character input.
  const cancelNewCharacter = () => {
    setNewCharacterName("");
    setNewCharacterDesc("");
    setShowNewCharacterInput(false);
  };

  // Function to call the backend and generate the story.
  const generateStory = async () => {
    setLoading(true);
    const payload = {
      beats: data.beats,
      characters: data.characters,
      setting: data.setting,
      genre: data.genre,
      style: data.style,
      temperature: temperature,
      approx_word_count: desiredLength,
      ...(coverStyle && { cover_style: coverStyle }),
      ...(visualPrompt.trim() && { visual_prompt: visualPrompt.trim() }),  
    };
    console.log("generateStory() payload:", payload);
    try {
      const res = await fetch("http://localhost:8000/generate-prose", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const result = await res.json();
      setGeneratedStory(result.prose_output);
      setCoverUrl(result.cover_image || null);
      setMessages(prev => [
        ...prev,
        { sender: 'bot', text: "Story generated. To view the full story, use the toggle button. Type 'start over' to reset." }
      ]);
      setCurrentPrompt("Type your next input:");
      setStep(STEPS.EXTEND);
      setLayoutMode("both");
    } catch (error) {
      console.error("Error generating story:", error);
      setMessages(prev => [
        ...prev,
        { sender: 'bot', text: "Error generating story. Please try again or type 'start over'." }
      ]);
      setCurrentPrompt("");
    }
    setLoading(false);
  };

  // Central Toggle Button handler.
  const cycleLayoutMode = () => {
    if (generatedStory === "") {
      setLayoutMode(prev => (prev === "both" ? "chatOnly" : "both"));
    } else {
      setLayoutMode(prev => (prev === "both" ? "storyOnly" : "both"));
    }
  };

  // Determine panel widths based on layout mode.
  let leftPanelWidth, rightPanelWidth;
  const beforeGeneration = generatedStory === "";
  if (beforeGeneration) {
    if (layoutMode === "chatOnly") {
      leftPanelWidth = 12;
      rightPanelWidth = 0;
    } else {
      leftPanelWidth = 9;
      rightPanelWidth = 3;
    }
  } else {
    if (layoutMode === "storyOnly") {
      leftPanelWidth = 0;
      rightPanelWidth = 12;
    } else {
      leftPanelWidth = 8;
      rightPanelWidth = 4;
    }
  }
  const chatVisible = beforeGeneration ? true : (layoutMode !== "storyOnly");
  const rightVisible = rightPanelWidth > 0;

  return (
    <Box
      sx={{
        width: "100vw",
        height: "100vh",
        p: 2,
        background: "#f0f2f5",
        overflow: "hidden",
        boxSizing: "border-box",
        position: "relative",
      }}
    >
      {/* Loading Overlay */}
      {loading && (
        <Box
          sx={{
            position: "absolute",
            top: 0,
            left: 0,
            width: "100%",
            height: "100%",
            background: "rgba(0,0,0,0.5)",
            zIndex: 200,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
          }}
        >
          <CircularProgress size={80} color="inherit" />
        </Box>
      )}

      {/* Central Toggle Button */}
      <Box
        sx={{
          position: "fixed",
          top: 16,
          right: 16,
          zIndex: 100,
        }}
      >
        <Button variant="contained" color="info" onClick={cycleLayoutMode} sx={{ fontSize: "0.9rem", px: 2 }}>
          {layoutMode === "both" ? "HIDE" : "SHOW"}
        </Button>
      </Box>

      <Grid container spacing={2} sx={{ height: "100%" }}>
        {/* Left Panel: Chat Panel */}
        {chatVisible && (
          <Grid item xs={leftPanelWidth} sx={{ height: "100%" }}>
            <Paper
              sx={{
                height: "100%",
                p: 2,
                borderRadius: 3,
                boxShadow: 3,
                background: "#fff",
                display: "flex",
                flexDirection: "column",
                overflow: "hidden",
              }}
            >
              <Typography variant="h5" sx={{ mb: 1, fontSize: "1.5rem", fontWeight: "bold" }}>
                Chat with ProseGenerator
              </Typography>
              <Box
                sx={{
                  flexGrow: 1,
                  display: 'flex',
                  flexDirection: 'column',
                  gap: 1,
                  overflowY: 'auto',
                  pr: 1,
                  mb: 1,
                }}
              >
                {messages.slice(-3).map((msg, index) => (
                  <Box key={index} sx={{ display: 'flex', justifyContent: msg.sender === 'bot' ? 'flex-start' : 'flex-end' }}>
                    <ChatBubble sender={msg.sender}>
                      <Typography variant="body1" sx={{ fontSize: "1.3rem" }}>
                        {msg.text}
                      </Typography>
                    </ChatBubble>
                  </Box>
                ))}
              </Box>
              <Box component="form" onSubmit={handleSend} sx={{ display: "flex" }}>
                <TextField
                  fullWidth
                  multiline
                  minRows={3}
                  variant="outlined"
                  value={userInput}
                  onChange={(e) => setUserInput(e.target.value)}
                  placeholder={currentPrompt}
                  sx={{ fontSize: "1.3rem", background: "#fff", borderRadius: 2 }}
                  InputProps={{ sx: { fontSize: "1.3rem", px: 2, py: 1 } }}
                  required
                />
                <Button type="submit" variant="contained" color="primary" sx={{ ml: 1, fontSize: "1.3rem", px: 3 }}>
                  Send
                </Button>
              </Box>
            </Paper>
          </Grid>
        )}

        {/* Right Panel: Parameters / Generated Story */}
        {rightVisible && (
          <Grid item xs={rightPanelWidth} sx={{ height: "100%", display: "flex" }}>
            <Paper
              sx={{
                height: "100%",
                p: 2,
                borderRadius: 3,
                boxShadow: 3,
                background: "#fff",
                display: "flex",
                flexDirection: "column",
                overflow: "hidden",
              }}
            >
              {beforeGeneration ? (
                <>
                  <Typography variant="h6" sx={{ mb: 1, fontSize: "1.3rem", fontWeight: "bold" }}>
                    Generation Parameters
                  </Typography>
                  {/* Sliders */}
                  <Box sx={{ mb: 2 }}>
                    <Typography variant="body1" sx={{ fontSize: "1.1rem" }}>
                      Temperature: {temperature.toFixed(2)}
                    </Typography>
                    <Slider
                      value={temperature}
                      onChange={(e, newValue) => setTemperature(newValue)}
                      min={0.0}
                      max={1.0}
                      step={0.05}
                      valueLabelDisplay="auto"
                    />
                  </Box>
                  <Box sx={{ mb: 2 }}>
                    <Typography variant="body1" sx={{ fontSize: "1.1rem" }}>
                      Desired Word Count: {desiredLength}
                    </Typography>
                    <Slider
                      value={desiredLength}
                      onChange={(e, newValue) => setDesiredLength(newValue)}
                      min={1000}
                      max={2500}
                      step={50}
                      valueLabelDisplay="auto"
                    />
                  </Box>
                  {/* Beats Section */}
                  <Box sx={{ mb: 2 }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                      <Typography variant="body2" sx={{ fontSize: "1.1rem", fontWeight: 'bold' }}>
                        Beats:
                      </Typography>
                      <IconButton onClick={() => setShowBeats((prev) => !prev)}>
                        {showBeats ? <ExpandLessIcon fontSize="small" /> : <ExpandMoreIcon fontSize="small" />}
                      </IconButton>
                      <IconButton onClick={() => setShowNewBeatInput((prev) => !prev)}>
                        <AddIcon fontSize="small" />
                      </IconButton>
                    </Box>
                    <Collapse in={showBeats}>
                      {data.beats.length > 0 && data.beats.map((beat, idx) => (
                        <Box key={idx} sx={{ display: 'flex', alignItems: 'center', mb: 0.5 }}>
                          <Typography variant="body2" sx={{ fontSize: "1.1rem", flexGrow: 1 }}>
                            {idx + 1}. {beat}
                          </Typography>
                          <IconButton size="small" onClick={() => removeBeat(idx)}>
                            <DeleteIcon fontSize="small" />
                          </IconButton>
                        </Box>
                      ))}
                      {showNewBeatInput && (
                        <Box sx={{ display: 'flex', mt: 1 }}>
                          <TextField
                            placeholder="Add new beat"
                            value={newBeat}
                            onChange={(e) => setNewBeat(e.target.value)}
                            size="small"
                            sx={{ flexGrow: 1, fontSize: "1.1rem" }}
                          />
                          <IconButton onClick={handleAddBeat} color="primary">
                            <CheckIcon fontSize="small" />
                          </IconButton>
                          <IconButton onClick={cancelNewBeat} color="error">
                            <DeleteIcon fontSize="small" />
                          </IconButton>
                        </Box>
                      )}
                    </Collapse>
                  </Box>
                  {/* Characters Section */}
                  <Box sx={{ mb: 2 }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                      <Typography variant="body2" sx={{ fontSize: "1.1rem", fontWeight: 'bold' }}>
                        Characters:
                      </Typography>
                      <IconButton onClick={() => setShowCharacters((prev) => !prev)}>
                        {showCharacters ? <ExpandLessIcon fontSize="small" /> : <ExpandMoreIcon fontSize="small" />}
                      </IconButton>
                      <IconButton onClick={() => setShowNewCharacterInput((prev) => !prev)}>
                        <AddIcon fontSize="small" />
                      </IconButton>
                    </Box>
                    <Collapse in={showCharacters}>
                      {data.characters.length > 0 && (
                        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                          {data.characters.map((char, idx) => (
                            <Box key={idx} sx={{ display: 'flex', alignItems: 'center' }}>
                              <Tooltip title={char.description} arrow>
                                <Button
                                  variant="contained"
                                  size="small"
                                  sx={{
                                    backgroundColor: char.color,
                                    textTransform: "none",
                                    fontSize: "0.9rem",
                                    '&:hover': { backgroundColor: char.color },
                                  }}
                                >
                                  {char.name}
                                </Button>
                              </Tooltip>
                              <IconButton size="small" onClick={() => removeCharacter(idx)}>
                                <DeleteIcon fontSize="small" />
                              </IconButton>
                            </Box>
                          ))}
                        </Box>
                      )}
                      {showNewCharacterInput && (
                        <Box sx={{ mt: 1, display: 'flex', gap: 1 }}>
                          <TextField
                            placeholder="Character Name"
                            value={newCharacterName}
                            onChange={(e) => setNewCharacterName(e.target.value)}
                            size="small"
                            sx={{ flexGrow: 1 }}
                          />
                          <TextField
                            placeholder="Description"
                            value={newCharacterDesc}
                            onChange={(e) => setNewCharacterDesc(e.target.value)}
                            size="small"
                            sx={{ flexGrow: 2 }}
                          />
                          <IconButton onClick={handleAddCharacter} color="primary">
                            <CheckIcon fontSize="small" />
                          </IconButton>
                          <IconButton onClick={cancelNewCharacter} color="error">
                            <DeleteIcon fontSize="small" />
                          </IconButton>
                        </Box>
                      )}
                    </Collapse>
                  </Box>
                  {/* Other Parameters Section */}
                  <Box sx={{ mb: 2 }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                      <Typography variant="body2" sx={{ fontSize: "1.1rem", fontWeight: 'bold' }}>
                        Other Parameters:
                      </Typography>
                      <IconButton onClick={() => setShowOtherParams((prev) => !prev)}>
                        {showOtherParams ? <ExpandLessIcon fontSize="small" /> : <ExpandMoreIcon fontSize="small" />}
                      </IconButton>
                    </Box>
                    <Collapse in={showOtherParams}>
                      <Box sx={{ mb: 1 }}>
                        <Typography variant="body2" sx={{ fontSize: "1.1rem", fontWeight: 'bold' }}>
                          Setting:
                        </Typography>
                        {editingSetting ? (
                          <Box sx={{ display: 'flex', alignItems: 'center' }}>
                            <TextField 
                              variant="outlined"
                              size="small"
                              value={tempSetting}
                              onChange={(e) => setTempSetting(e.target.value)}
                              sx={{ flexGrow: 1 }}
                            />
                            <IconButton onClick={() => saveEdit("setting")}>
                              <CheckIcon fontSize="small" />
                            </IconButton>
                          </Box>
                        ) : (
                          <Box sx={{ display: 'flex', alignItems: 'center' }}>
                            <Typography variant="body2" sx={{ flexGrow: 1 }}>
                              {data.setting}
                            </Typography>
                            <IconButton onClick={() => toggleEdit("setting")}>
                              <EditIcon fontSize="small" />
                            </IconButton>
                          </Box>
                        )}
                      </Box>
                      <Box sx={{ mb: 1 }}>
                        <Typography variant="body2" sx={{ fontSize: "1.1rem", fontWeight: 'bold' }}>
                          Genre:
                        </Typography>
                        {editingGenre ? (
                          <Box sx={{ display: 'flex', alignItems: 'center' }}>
                            <TextField 
                              variant="outlined"
                              size="small"
                              value={tempGenre}
                              onChange={(e) => setTempGenre(e.target.value)}
                              sx={{ flexGrow: 1 }}
                            />
                            <IconButton onClick={() => saveEdit("genre")}>
                              <CheckIcon fontSize="small" />
                            </IconButton>
                          </Box>
                        ) : (
                          <Box sx={{ display: 'flex', alignItems: 'center' }}>
                            <Typography variant="body2" sx={{ flexGrow: 1 }}>
                              {data.genre}
                            </Typography>
                            <IconButton onClick={() => toggleEdit("genre")}>
                              <EditIcon fontSize="small" />
                            </IconButton>
                          </Box>
                        )}
                      </Box>
                      <Box sx={{ mb: 1 }}>
                        <Typography variant="body2" sx={{ fontSize: "1.1rem", fontWeight: 'bold' }}>
                          Style:
                        </Typography>
                        {editingStyle ? (
                          <Box sx={{ display: 'flex', alignItems: 'center' }}>
                            <TextField 
                              variant="outlined"
                              size="small"
                              value={tempStyle}
                              onChange={(e) => setTempStyle(e.target.value)}
                              sx={{ flexGrow: 1 }}
                            />
                            <IconButton onClick={() => saveEdit("style")}>
                              <CheckIcon fontSize="small" />
                            </IconButton>
                          </Box>
                        ) : (
                          <Box sx={{ display: 'flex', alignItems: 'center' }}>
                            <Typography variant="body2" sx={{ flexGrow: 1 }}>
                              {data.style}
                            </Typography>
                            <IconButton onClick={() => toggleEdit("style")}>
                              <EditIcon fontSize="small" />
                            </IconButton>
                          </Box>
                        )}
                        {/* Cover-art Style */}
                          <Box sx={{ mb: 1 }}>
                            <Typography variant="body2" sx={{ fontSize: "1.1rem", fontWeight: 'bold' }}>
                              Cover-art style:
                            </Typography>
                            <TextField
                              select
                              SelectProps={{ native: true }}
                              size="small"
                              value={coverStyle}
                              onChange={(e) => setCoverStyle(e.target.value)}
                              sx={{ mt: 0.5, width: '100%' }}
                            >
                              {COVER_OPTIONS.map(opt => (
                                <option key={opt.key} value={opt.key}>{opt.label}</option>
                              ))}
                            </TextField>
                          </Box>
                          {/* Custom image prompt — user can override or add hints */}
<Box sx={{ mb: 1 }}>
                        <Typography variant="body2" sx={{ fontSize: "1.1rem", fontWeight: 'bold' }}>
                          Image prompt hints:
                        </Typography>

                        <TextField
                          fullWidth
                          size="small"
                          placeholder="e.g. neon dusk, fog, teal-orange contrast"
                          value={visualPrompt}
                          onChange={(e) => setVisualPrompt(e.target.value)}
                          sx={{ mt: 0.5 }}
                        />
                      </Box>
                      </Box>
                    </Collapse>
                  </Box>
                  <Typography variant="body2" sx={{ fontSize: "1.1rem" }}>
                    Adjust the parameters above before generating your story.
                  </Typography>
                </>
              ) : (
                <>
                  <Typography variant="h6" sx={{ mb: 1, fontSize: "1.3rem", fontWeight: "bold" }}>
                    Generated Story
                  </Typography>
                  <Divider sx={{ mb: 2 }} />
                  {coverUrl && (
                  <Box sx={{ mb: 2, textAlign: 'center' }}>
                                      <img
                    src={`http://localhost:8000${coverUrl}`}
                    alt="Cover art"
                    style={{
                      width: '100%',
                      maxWidth: 400,
                      aspectRatio: '1 / 1',
                      objectFit: 'cover',
                      borderRadius: 8,
                      boxShadow: '0 2px 6px rgba(0,0,0,0.15)',
                    }}
                  />

                  </Box>
                )}
                  <Box
                    sx={{
                      display: 'flex',
                      flexDirection: 'column',
                      flexGrow: 1,
                      overflowY: 'auto',
                      pr: 1,
                      pb: 4,
                    }}
                  >
                    <Typography
                      variant="body1"
                      sx={{
                        fontSize: "1.1rem",
                        whiteSpace: "pre-line",
                        lineHeight: 1.6,
                      }}
                      dangerouslySetInnerHTML={{ __html: highlightStory(generatedStory) }}
                    />
                  </Box>
                </>
              )}
            </Paper>
          </Grid>
        )}
      </Grid>
    </Box>
  );
}