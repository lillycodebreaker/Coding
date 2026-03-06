import { useState, useRef, useCallback } from "react";

const STYLE_PROFILES = ["Casual Chic", "Business Professional", "Streetwear", "Bohemian", "Minimalist", "Date Night"];

const CLAUDE_SYSTEM_PROMPT = `You are VOGUE.AI, an elite fashion stylist with expertise in personal styling, color theory, and current trends. 

When given a photo of someone, analyze:
1. Their apparent body shape, coloring, and current outfit
2. What styles and cuts would flatter them
3. Specific clothing recommendations

Return ONLY valid JSON in this exact format (no markdown, no extra text):
{
  "style_analysis": "2-3 sentences analyzing their current look and personal style vibe",
  "color_palette": ["color1", "color2", "color3", "color4"],
  "body_shape": "brief body shape description",
  "recommendations": [
    {
      "category": "Tops",
      "item": "specific item name",
      "description": "why this works for them",
      "amazon_search": "exact search query for Amazon",
      "style_tip": "how to wear it"
    },
    {
      "category": "Bottoms", 
      "item": "specific item name",
      "description": "why this works for them",
      "amazon_search": "exact search query for Amazon",
      "style_tip": "how to wear it"
    },
    {
      "category": "Outerwear",
      "item": "specific item name", 
      "description": "why this works for them",
      "amazon_search": "exact search query for Amazon",
      "style_tip": "how to wear it"
    },
    {
      "category": "Footwear",
      "item": "specific item name",
      "description": "why this works for them",
      "amazon_search": "exact search query for Amazon",
      "style_tip": "how to wear it"
    },
    {
      "category": "Accessories",
      "item": "specific item name",
      "description": "why this works for them",
      "amazon_search": "exact search query for Amazon",
      "style_tip": "how to wear it"
    },
    {
      "category": "Complete Outfit",
      "item": "full outfit combination",
      "description": "how the pieces work together",
      "amazon_search": "exact search query for Amazon",
      "style_tip": "the complete look"
    }
  ],
  "overall_tip": "One powerful personal styling tip for this person"
}`;

const CATEGORY_ICONS = {
  "Tops": "👚",
  "Bottoms": "👖",
  "Outerwear": "🧥",
  "Footwear": "👟",
  "Accessories": "💍",
  "Complete Outfit": "✨"
};

export default function FashionRecommender() {
  const [image, setImage] = useState(null);
  const [imageBase64, setImageBase64] = useState(null);
  const [styleProfile, setStyleProfile] = useState("Casual Chic");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const [activeCard, setActiveCard] = useState(null);
  const fileInputRef = useRef(null);

  const processFile = useCallback((file) => {
    if (!file || !file.type.startsWith("image/")) return;
    const url = URL.createObjectURL(file);
    setImage(url);
    setResult(null);
    setError(null);
    const reader = new FileReader();
    reader.onload = (e) => {
      const base64 = e.target.result.split(",")[1];
      setImageBase64(base64);
    };
    reader.readAsDataURL(file);
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    processFile(file);
  }, [processFile]);

  const handleFileChange = (e) => processFile(e.target.files[0]);

  const getAmazonLink = (searchQuery) => {
    const encoded = encodeURIComponent(searchQuery + " women fashion");
    return `https://www.amazon.com/s?k=${encoded}&ref=fashion_ai`;
  };

  const analyzeOutfit = async () => {
    if (!imageBase64) return;
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch("https://api.anthropic.com/v1/messages", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: "claude-sonnet-4-20250514",
          max_tokens: 1000,
          system: CLAUDE_SYSTEM_PROMPT,
          messages: [{
            role: "user",
            content: [
              {
                type: "image",
                source: { type: "base64", media_type: "image/jpeg", data: imageBase64 }
              },
              {
                type: "text",
                text: `Analyze this person's style and give personalized fashion recommendations optimized for a ${styleProfile} aesthetic. Return ONLY the JSON object, no other text.`
              }
            ]
          }]
        })
      });

      const data = await response.json();
      const text = data.content?.map(b => b.text || "").join("") || "";
      const clean = text.replace(/```json|```/g, "").trim();
      const parsed = JSON.parse(clean);
      setResult(parsed);
    } catch (err) {
      setError("Could not analyze your style. Please try again with a clearer photo.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{
      minHeight: "100vh",
      background: "#0a0a0a",
      fontFamily: "'Georgia', serif",
      color: "#f0ece4",
      overflowX: "hidden"
    }}>
      {/* Header */}
      <header style={{
        borderBottom: "1px solid #2a2a2a",
        padding: "24px 40px",
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        background: "rgba(10,10,10,0.95)",
        position: "sticky",
        top: 0,
        zIndex: 100,
        backdropFilter: "blur(12px)"
      }}>
        <div>
          <div style={{
            fontSize: "11px",
            letterSpacing: "0.4em",
            color: "#c9a96e",
            textTransform: "uppercase",
            marginBottom: "2px"
          }}>AI-Powered</div>
          <div style={{
            fontSize: "28px",
            fontWeight: "300",
            letterSpacing: "0.15em",
            color: "#f0ece4"
          }}>STYLE<span style={{ color: "#c9a96e", fontStyle: "italic" }}>ORACLE</span></div>
        </div>
        <div style={{
          fontSize: "11px",
          color: "#666",
          letterSpacing: "0.2em",
          textTransform: "uppercase"
        }}>Personal Fashion Intelligence</div>
      </header>

      <main style={{ maxWidth: "1100px", margin: "0 auto", padding: "60px 40px" }}>

        {/* Hero */}
        <div style={{ textAlign: "center", marginBottom: "64px" }}>
          <h1 style={{
            fontSize: "clamp(36px, 6vw, 72px)",
            fontWeight: "300",
            letterSpacing: "0.05em",
            lineHeight: 1.1,
            marginBottom: "20px",
            color: "#f0ece4"
          }}>
            Dress Like<br />
            <em style={{ color: "#c9a96e", fontStyle: "italic" }}>Your Best Self</em>
          </h1>
          <p style={{ fontSize: "16px", color: "#888", letterSpacing: "0.05em", maxWidth: "480px", margin: "0 auto" }}>
            Upload a photo. Get curated outfit recommendations with direct Amazon shopping links.
          </p>
        </div>

        {/* Upload + Config */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "32px", marginBottom: "48px" }}>

          {/* Upload Zone */}
          <div
            onDrop={handleDrop}
            onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
            onDragLeave={() => setDragOver(false)}
            onClick={() => fileInputRef.current?.click()}
            style={{
              border: `2px dashed ${dragOver ? "#c9a96e" : image ? "#c9a96e" : "#333"}`,
              borderRadius: "4px",
              padding: "40px",
              cursor: "pointer",
              transition: "all 0.3s ease",
              background: dragOver ? "rgba(201,169,110,0.05)" : "rgba(255,255,255,0.02)",
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              justifyContent: "center",
              minHeight: "300px",
              position: "relative",
              overflow: "hidden"
            }}
          >
            {image ? (
              <>
                <img src={image} alt="Your photo" style={{
                  maxWidth: "100%",
                  maxHeight: "260px",
                  objectFit: "contain",
                  borderRadius: "2px"
                }} />
                <div style={{
                  position: "absolute",
                  bottom: "12px",
                  fontSize: "11px",
                  color: "#c9a96e",
                  letterSpacing: "0.2em",
                  textTransform: "uppercase"
                }}>Click to change</div>
              </>
            ) : (
              <>
                <div style={{ fontSize: "48px", marginBottom: "16px", opacity: 0.4 }}>📸</div>
                <div style={{ fontSize: "14px", color: "#888", letterSpacing: "0.1em", textAlign: "center" }}>
                  Drop your photo here<br />
                  <span style={{ color: "#555", fontSize: "12px" }}>or click to browse</span>
                </div>
              </>
            )}
            <input ref={fileInputRef} type="file" accept="image/*" onChange={handleFileChange} style={{ display: "none" }} />
          </div>

          {/* Config Panel */}
          <div style={{
            background: "rgba(255,255,255,0.02)",
            border: "1px solid #222",
            borderRadius: "4px",
            padding: "36px"
          }}>
            <div style={{
              fontSize: "10px",
              letterSpacing: "0.35em",
              color: "#c9a96e",
              textTransform: "uppercase",
              marginBottom: "24px"
            }}>Style Direction</div>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "8px", marginBottom: "32px" }}>
              {STYLE_PROFILES.map(style => (
                <button
                  key={style}
                  onClick={() => setStyleProfile(style)}
                  style={{
                    padding: "10px 12px",
                    background: styleProfile === style ? "#c9a96e" : "transparent",
                    border: `1px solid ${styleProfile === style ? "#c9a96e" : "#333"}`,
                    borderRadius: "2px",
                    color: styleProfile === style ? "#0a0a0a" : "#888",
                    fontSize: "11px",
                    letterSpacing: "0.1em",
                    cursor: "pointer",
                    transition: "all 0.2s ease",
                    fontFamily: "inherit"
                  }}
                >
                  {style}
                </button>
              ))}
            </div>

            <div style={{
              fontSize: "12px",
              color: "#555",
              lineHeight: 1.6,
              marginBottom: "32px",
              padding: "16px",
              borderLeft: "2px solid #222"
            }}>
              Selected: <span style={{ color: "#c9a96e" }}>{styleProfile}</span>
              <br />Our AI will tailor recommendations to match this aesthetic perfectly.
            </div>

            <button
              onClick={analyzeOutfit}
              disabled={!imageBase64 || loading}
              style={{
                width: "100%",
                padding: "16px",
                background: imageBase64 && !loading ? "#c9a96e" : "#1a1a1a",
                border: "none",
                borderRadius: "2px",
                color: imageBase64 && !loading ? "#0a0a0a" : "#444",
                fontSize: "12px",
                letterSpacing: "0.3em",
                textTransform: "uppercase",
                cursor: imageBase64 && !loading ? "pointer" : "not-allowed",
                transition: "all 0.3s ease",
                fontFamily: "inherit",
                fontWeight: "600"
              }}
            >
              {loading ? "Analyzing Your Style..." : "Get My Recommendations →"}
            </button>
          </div>
        </div>

        {/* Loading State */}
        {loading && (
          <div style={{ textAlign: "center", padding: "80px 0" }}>
            <div style={{
              width: "48px",
              height: "48px",
              border: "1px solid #333",
              borderTop: "1px solid #c9a96e",
              borderRadius: "50%",
              animation: "spin 1s linear infinite",
              margin: "0 auto 24px"
            }} />
            <div style={{ fontSize: "12px", letterSpacing: "0.3em", color: "#666", textTransform: "uppercase" }}>
              Consulting your stylist...
            </div>
            <style>{`@keyframes spin { to { transform: rotate(360deg) } }`}</style>
          </div>
        )}

        {/* Error */}
        {error && (
          <div style={{
            border: "1px solid #5a2020",
            background: "rgba(90,32,32,0.1)",
            padding: "20px 24px",
            borderRadius: "2px",
            fontSize: "14px",
            color: "#e88",
            marginBottom: "32px"
          }}>
            {error}
          </div>
        )}

        {/* Results */}
        {result && (
          <div style={{ animation: "fadeIn 0.6s ease" }}>
            <style>{`
              @keyframes fadeIn { from { opacity: 0; transform: translateY(20px) } to { opacity: 1; transform: translateY(0) } }
              .rec-card:hover { border-color: #c9a96e !important; transform: translateY(-2px); }
              .amazon-btn:hover { background: #c9a96e !important; color: #0a0a0a !important; }
            `}</style>

            {/* Style Analysis */}
            <div style={{
              background: "linear-gradient(135deg, rgba(201,169,110,0.08) 0%, rgba(201,169,110,0.02) 100%)",
              border: "1px solid rgba(201,169,110,0.3)",
              borderRadius: "4px",
              padding: "36px",
              marginBottom: "48px"
            }}>
              <div style={{ fontSize: "10px", letterSpacing: "0.35em", color: "#c9a96e", textTransform: "uppercase", marginBottom: "16px" }}>
                Style Analysis
              </div>
              <p style={{ fontSize: "16px", lineHeight: 1.8, color: "#d4cfc8", marginBottom: "24px" }}>
                {result.style_analysis}
              </p>

              <div style={{ display: "flex", gap: "32px", flexWrap: "wrap" }}>
                <div>
                  <div style={{ fontSize: "10px", letterSpacing: "0.3em", color: "#666", textTransform: "uppercase", marginBottom: "10px" }}>
                    Your Color Palette
                  </div>
                  <div style={{ display: "flex", gap: "8px" }}>
                    {result.color_palette?.map((color, i) => (
                      <div key={i} style={{
                        width: "36px",
                        height: "36px",
                        borderRadius: "50%",
                        background: color,
                        border: "2px solid rgba(255,255,255,0.1)",
                        title: color
                      }} title={color} />
                    ))}
                  </div>
                </div>
                <div>
                  <div style={{ fontSize: "10px", letterSpacing: "0.3em", color: "#666", textTransform: "uppercase", marginBottom: "10px" }}>
                    Body Shape
                  </div>
                  <div style={{ fontSize: "14px", color: "#c9a96e" }}>{result.body_shape}</div>
                </div>
              </div>
            </div>

            {/* Overall Tip */}
            {result.overall_tip && (
              <div style={{
                borderLeft: "3px solid #c9a96e",
                paddingLeft: "24px",
                marginBottom: "48px",
                fontStyle: "italic",
                fontSize: "16px",
                color: "#a09890",
                lineHeight: 1.7
              }}>
                "{result.overall_tip}"
              </div>
            )}

            {/* Recommendation Cards */}
            <div style={{ fontSize: "10px", letterSpacing: "0.35em", color: "#c9a96e", textTransform: "uppercase", marginBottom: "24px" }}>
              Curated Picks for You
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(300px, 1fr))", gap: "16px" }}>
              {result.recommendations?.map((rec, i) => (
                <div
                  key={i}
                  className="rec-card"
                  onClick={() => setActiveCard(activeCard === i ? null : i)}
                  style={{
                    background: "rgba(255,255,255,0.02)",
                    border: "1px solid #222",
                    borderRadius: "4px",
                    padding: "24px",
                    cursor: "pointer",
                    transition: "all 0.25s ease"
                  }}
                >
                  <div style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "12px" }}>
                    <span style={{ fontSize: "22px" }}>{CATEGORY_ICONS[rec.category] || "🛍️"}</span>
                    <span style={{ fontSize: "10px", letterSpacing: "0.3em", color: "#c9a96e", textTransform: "uppercase" }}>
                      {rec.category}
                    </span>
                  </div>

                  <div style={{ fontSize: "17px", fontWeight: "400", marginBottom: "8px", color: "#f0ece4" }}>
                    {rec.item}
                  </div>

                  <p style={{ fontSize: "13px", color: "#888", lineHeight: 1.6, marginBottom: "16px" }}>
                    {rec.description}
                  </p>

                  {activeCard === i && (
                    <div style={{
                      borderTop: "1px solid #222",
                      paddingTop: "14px",
                      marginBottom: "16px",
                      fontSize: "12px",
                      color: "#a09890",
                      lineHeight: 1.6,
                      fontStyle: "italic"
                    }}>
                      💡 {rec.style_tip}
                    </div>
                  )}

                  <a
                    href={getAmazonLink(rec.amazon_search)}
                    target="_blank"
                    rel="noopener noreferrer"
                    onClick={(e) => e.stopPropagation()}
                    className="amazon-btn"
                    style={{
                      display: "block",
                      textAlign: "center",
                      padding: "10px",
                      border: "1px solid #c9a96e",
                      borderRadius: "2px",
                      color: "#c9a96e",
                      fontSize: "10px",
                      letterSpacing: "0.3em",
                      textTransform: "uppercase",
                      textDecoration: "none",
                      transition: "all 0.2s ease"
                    }}
                  >
                    Shop on Amazon →
                  </a>
                </div>
              ))}
            </div>

            {/* Restart */}
            <div style={{ textAlign: "center", marginTop: "64px" }}>
              <button
                onClick={() => { setResult(null); setImage(null); setImageBase64(null); }}
                style={{
                  background: "transparent",
                  border: "1px solid #333",
                  color: "#666",
                  padding: "12px 32px",
                  fontSize: "11px",
                  letterSpacing: "0.3em",
                  textTransform: "uppercase",
                  cursor: "pointer",
                  borderRadius: "2px",
                  fontFamily: "inherit",
                  transition: "all 0.2s ease"
                }}
                onMouseOver={(e) => { e.target.style.borderColor = "#c9a96e"; e.target.style.color = "#c9a96e"; }}
                onMouseOut={(e) => { e.target.style.borderColor = "#333"; e.target.style.color = "#666"; }}
              >
                Try Another Photo
              </button>
            </div>
          </div>
        )}
      </main>

      <footer style={{
        marginTop: "80px",
        padding: "32px 40px",
        borderTop: "1px solid #1a1a1a",
        textAlign: "center",
        fontSize: "11px",
        color: "#444",
        letterSpacing: "0.2em"
      }}>
        STYLEORACLE — AI-POWERED PERSONAL STYLING
      </footer>
    </div>
  );
}
