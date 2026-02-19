import { useState, useEffect, useRef } from "react";

// ─── Embedded Talent Databases ─────────────────────────────────────────────
const DIRECTORS = [
  { name: "Christopher Nolan", avg_roi: 3.2, avg_rev_mult: 3.8, success_rate: 0.87 },
  { name: "Steven Spielberg", avg_roi: 2.8, avg_rev_mult: 3.4, success_rate: 0.82 },
  { name: "James Cameron", avg_roi: 3.5, avg_rev_mult: 4.1, success_rate: 0.89 },
  { name: "Quentin Tarantino", avg_roi: 2.4, avg_rev_mult: 3.0, success_rate: 0.78 },
  { name: "Martin Scorsese", avg_roi: 2.1, avg_rev_mult: 2.7, success_rate: 0.74 },
  { name: "Denis Villeneuve", avg_roi: 2.9, avg_rev_mult: 3.5, success_rate: 0.84 },
  { name: "Jon Favreau", avg_roi: 3.1, avg_rev_mult: 3.9, success_rate: 0.86 },
  { name: "Russo Brothers", avg_roi: 3.4, avg_rev_mult: 4.0, success_rate: 0.88 },
  { name: "Joss Whedon", avg_roi: 2.6, avg_rev_mult: 3.2, success_rate: 0.79 },
  { name: "Ridley Scott", avg_roi: 2.2, avg_rev_mult: 2.8, success_rate: 0.75 },
  { name: "David Fincher", avg_roi: 1.9, avg_rev_mult: 2.4, success_rate: 0.71 },
  { name: "Greta Gerwig", avg_roi: 2.7, avg_rev_mult: 3.3, success_rate: 0.81 },
];

const PRODUCERS = [
  { name: "Jerry Bruckheimer", avg_roi: 2.9, avg_rev_mult: 3.6, quality: 0.81 },
  { name: "Kevin Feige", avg_roi: 3.3, avg_rev_mult: 4.0, quality: 0.88 },
  { name: "Neal H. Moritz", avg_roi: 2.8, avg_rev_mult: 3.4, quality: 0.79 },
  { name: "Frank Marshall", avg_roi: 2.5, avg_rev_mult: 3.1, quality: 0.76 },
  { name: "Jerry Weintraub", avg_roi: 2.3, avg_rev_mult: 2.9, quality: 0.74 },
  { name: "Barbara Broccoli", avg_roi: 2.6, avg_rev_mult: 3.2, quality: 0.78 },
  { name: "Jon Landau", avg_roi: 3.0, avg_rev_mult: 3.7, quality: 0.83 },
  { name: "Stan Lee", avg_roi: 3.2, avg_rev_mult: 3.9, quality: 0.86 },
  { name: "Richard Zanuck", avg_roi: 2.4, avg_rev_mult: 3.0, quality: 0.75 },
  { name: "Kathleen Kennedy", avg_roi: 2.7, avg_rev_mult: 3.3, quality: 0.80 },
];

const WRITERS = [
  { name: "Aaron Sorkin", avg_roi: 2.1, avg_rev_mult: 2.6, quality: 0.82 },
  { name: "Christopher McQuarrie", avg_roi: 2.8, avg_rev_mult: 3.4, quality: 0.85 },
  { name: "Jonathan Nolan", avg_roi: 3.0, avg_rev_mult: 3.6, quality: 0.87 },
  { name: "Sylvester Stallone", avg_roi: 2.5, avg_rev_mult: 3.1, quality: 0.78 },
  { name: "James Cameron", avg_roi: 3.2, avg_rev_mult: 3.8, quality: 0.88 },
  { name: "Lilly Wachowski", avg_roi: 2.2, avg_rev_mult: 2.7, quality: 0.79 },
  { name: "David S. Goyer", avg_roi: 2.6, avg_rev_mult: 3.2, quality: 0.80 },
  { name: "Andrew Niccol", avg_roi: 2.0, avg_rev_mult: 2.5, quality: 0.77 },
  { name: "Dan Gilroy", avg_roi: 1.8, avg_rev_mult: 2.3, quality: 0.75 },
  { name: "Taika Waititi", avg_roi: 2.4, avg_rev_mult: 3.0, quality: 0.81 },
];

// ─── Genre and Month Data ──────────────────────────────────────────────────
const GENRE_MULTIPLIERS = {
  Action: 1.15,
  Comedy: 1.0,
  Drama: 0.85,
  Horror: 1.05,
  "Sci-Fi": 1.25,
  Thriller: 1.05,
  Animation: 1.10,
  Romance: 0.92,
  Fantasy: 1.20,
  Adventure: 1.18,
};

const MONTH_MULTIPLIERS = {
  1: 0.95,
  2: 0.92,
  3: 0.98,
  4: 1.00,
  5: 1.05,
  6: 1.08,
  7: 1.12,
  8: 1.10,
  9: 1.02,
  10: 1.00,
  11: 1.08,
  12: 1.15,
};

const MONTH_NAMES = [
  "",
  "January",
  "February",
  "March",
  "April",
  "May",
  "June",
  "July",
  "August",
  "September",
  "October",
  "November",
  "December",
];

// ─── Color Palette ────────────────────────────────────────────────────────
const COLORS = {
  navy: "#0f172a",
  navyLight: "#1e293b",
  navyMid: "#334155",
  slate: "#475569",
  slateLight: "#94a3b8",
  teal: "#06b6d4",
  tealDark: "#0891b2",
  tealGlow: "rgba(6, 182, 212, 0.15)",
  emerald: "#10b981",
  amber: "#f59e0b",
  rose: "#f43f5e",
  violet: "#8b5cf6",
  white: "#f8fafc",
};

const BAR_COLORS = [
  "linear-gradient(90deg, #06b6d4, #22d3ee)",
  "linear-gradient(90deg, #8b5cf6, #a78bfa)",
  "linear-gradient(90deg, #10b981, #34d399)",
  "linear-gradient(90deg, #f59e0b, #fbbf24)",
  "linear-gradient(90deg, #f43f5e, #fb7185)",
  "linear-gradient(90deg, #6366f1, #818cf8)",
];

// ─── Utility Functions ────────────────────────────────────────────────────
function formatMoney(n) {
  if (n >= 1_000_000_000) return `$${(n / 1_000_000_000).toFixed(1)}B`;
  if (n >= 1_000_000) return `$${(n / 1_000_000).toFixed(0)}M`;
  if (n >= 1_000) return `$${(n / 1_000).toFixed(0)}K`;
  return `$${n}`;
}

function getBudgetTier(budget) {
  if (budget < 5_000_000) return "Micro";
  if (budget < 20_000_000) return "Low";
  if (budget < 75_000_000) return "Mid";
  if (budget < 150_000_000) return "High";
  return "Blockbuster";
}

function getRoiColor(roi) {
  if (roi > 100) return COLORS.emerald;
  if (roi > 0) return COLORS.teal;
  if (roi > -30) return COLORS.amber;
  return COLORS.rose;
}

// ─── Client-Side Prediction Engine ─────────────────────────────────────────
function predictMovieSuccess(input) {
  const director = DIRECTORS.find(d => d.name === input.director) || DIRECTORS[0];
  const producer = PRODUCERS.find(p => p.name === input.producer) || PRODUCERS[0];
  const writer = WRITERS.find(w => w.name === input.writer) || WRITERS[0];

  const genreMultiplier = GENRE_MULTIPLIERS[input.genre] || 1.0;
  const monthMultiplier = MONTH_MULTIPLIERS[input.month] || 1.0;
  const budgetTier = getBudgetTier(input.budget);

  // Base revenue calculation
  const baseRevenuePerBudget = {
    Micro: 2.5,
    Low: 2.8,
    Mid: 3.2,
    High: 3.5,
    Blockbuster: 3.8,
  };

  const baseMult = baseRevenuePerBudget[budgetTier] || 3.0;

  // Calculate weighted talent score
  const talentScore =
    director.avg_rev_mult * 0.5 +
    (producer.avg_rev_mult || 2.5) * 0.25 +
    (writer.avg_rev_mult || 2.5) * 0.25;

  // Combined multiplier
  const combinedMult =
    baseMult * talentScore * genreMultiplier * monthMultiplier;

  // Add variation based on talent
  const successRateAvg =
    (director.success_rate + (producer.quality || 0.8) + (writer.quality || 0.8)) / 3;
  const variation = 0.95 + successRateAvg * 0.1;

  const predictedRevenue = Math.round(input.budget * combinedMult * variation);
  const roiPercent = Math.round(((predictedRevenue - input.budget) / input.budget) * 100);

  // Revenue range (±15% confidence interval)
  const revenueLow = Math.round(predictedRevenue * 0.85);
  const revenueHigh = Math.round(predictedRevenue * 1.15);

  // Success probability (threshold: 1.5x budget)
  const threshold = input.budget * 1.5;
  const zScore = (Math.log(predictedRevenue) - Math.log(threshold)) / 0.5;
  const successProbability = Math.round(
    (1 / (1 + Math.exp(-zScore))) * 100
  );

  // Confidence level
  let confidence = "Low";
  if (successRateAvg > 0.85) confidence = "High";
  else if (successRateAvg > 0.75) confidence = "Medium";

  // Feature importance
  const featureImportance = {
    Director: Math.round(director.avg_roi * 20),
    Genre: Math.round(genreMultiplier * 25),
    Budget: Math.round(baseMult * 18),
    "Release Month": Math.round(monthMultiplier * 12),
    Producer: Math.round((producer.avg_roi || 2.5) * 15),
    Writer: Math.round((writer.avg_roi || 2.0) * 10),
  };

  // Normalize to 100%
  const total = Object.values(featureImportance).reduce((a, b) => a + b, 1);
  Object.keys(featureImportance).forEach(key => {
    featureImportance[key] = Math.round((featureImportance[key] / total) * 100);
  });

  return {
    predicted_revenue: predictedRevenue,
    revenue_low: revenueLow,
    revenue_high: revenueHigh,
    roi_percent: roiPercent,
    budget: input.budget,
    success_probability: successProbability,
    confidence,
    feature_importance: featureImportance,
    input: {
      director: input.director,
      producer: input.producer || "Unknown",
      writer: input.writer || "Unknown",
      genre: input.genre,
      budget: input.budget,
      release_month: input.month,
    },
  };
}

// ─── SearchableSelect Component ────────────────────────────────────────────
function SearchableSelect({ label, required, options, placeholder, value, onChange }) {
  const [query, setQuery] = useState("");
  const [isOpen, setIsOpen] = useState(false);
  const [inputText, setInputText] = useState(value || "");
  const wrapperRef = useRef(null);

  const filtered = options.filter(opt =>
    opt.name.toLowerCase().includes(query.toLowerCase())
  );

  useEffect(() => {
    function handleClickOutside(e) {
      if (wrapperRef.current && !wrapperRef.current.contains(e.target)) {
        setIsOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  function handleSelect(item) {
    setInputText(item.name);
    setQuery("");
    setIsOpen(false);
    onChange(item.name);
  }

  function getStatValue(item) {
    if (item.avg_roi !== undefined) return `ROI: ${item.avg_roi}x`;
    if (item.avg_rev_mult !== undefined) return `Rev: ${item.avg_rev_mult}x`;
    if (item.quality !== undefined) return `Quality: ${(item.quality * 100).toFixed(0)}%`;
    return "";
  }

  return (
    <div style={{ marginBottom: "18px" }}>
      <label
        style={{
          display: "block",
          fontSize: "12px",
          fontWeight: 600,
          color: COLORS.slateLight,
          textTransform: "uppercase",
          letterSpacing: "0.5px",
          marginBottom: "6px",
        }}
      >
        {label} {required && <span style={{ color: COLORS.rose }}>*</span>}
      </label>
      <div style={{ position: "relative" }} ref={wrapperRef}>
        <input
          type="text"
          placeholder={placeholder}
          value={inputText}
          onChange={e => {
            setInputText(e.target.value);
            setQuery(e.target.value);
            setIsOpen(true);
            if (!e.target.value) onChange("");
          }}
          onFocus={() => setIsOpen(true)}
          style={{
            width: "100%",
            background: COLORS.navy,
            border: `1px solid ${COLORS.slate}`,
            borderRadius: "8px",
            padding: "10px 14px",
            color: COLORS.white,
            fontSize: "14px",
            fontFamily: "inherit",
            outline: "none",
            transition: "border-color 0.2s",
            boxSizing: "border-box",
          }}
          onFocus={e => {
            e.target.style.borderColor = COLORS.teal;
            e.target.style.boxShadow = `0 0 0 3px ${COLORS.tealGlow}`;
          }}
          onBlur={e => {
            e.target.style.boxShadow = "none";
          }}
        />
        {isOpen && filtered.length > 0 && (
          <div
            style={{
              position: "absolute",
              top: "calc(100% + 4px)",
              left: 0,
              right: 0,
              background: COLORS.navyLight,
              border: `1px solid ${COLORS.slate}`,
              borderRadius: "8px",
              maxHeight: "200px",
              overflowY: "auto",
              zIndex: 100,
              boxShadow: "0 8px 32px rgba(0,0,0,0.5)",
            }}
          >
            {filtered.map((item, i) => (
              <div
                key={i}
                onClick={() => handleSelect(item)}
                style={{
                  padding: "10px 14px",
                  cursor: "pointer",
                  fontSize: "13px",
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                  transition: "background 0.15s",
                  background: "transparent",
                }}
                onMouseEnter={e => {
                  e.currentTarget.style.background = COLORS.navyMid;
                }}
                onMouseLeave={e => {
                  e.currentTarget.style.background = "transparent";
                }}
              >
                <span style={{ color: COLORS.white, fontWeight: 500 }}>
                  {item.name}
                </span>
                <span style={{ color: COLORS.teal, fontSize: "11px", fontWeight: 600 }}>
                  {getStatValue(item)}
                </span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

// ─── Gauge Component ──────────────────────────────────────────────────────
function Gauge({ value }) {
  const pct = Math.max(0, Math.min(100, value));
  const r = 70;
  const cx = 90;
  const cy = 88;

  function polarToCart(angleDeg) {
    const rad = (angleDeg - 180) * Math.PI / 180;
    return { x: cx + r * Math.cos(rad), y: cy + r * Math.sin(rad) };
  }

  const startAngle = -180;
  const endAngle = startAngle + (pct / 100) * 180;
  const start = polarToCart(startAngle);
  const end = polarToCart(endAngle);
  const largeArc = pct > 50 ? 1 : 0;

  const color = pct >= 60 ? COLORS.emerald : pct >= 35 ? COLORS.amber : COLORS.rose;
  const label = pct >= 60 ? "Low Risk" : pct >= 35 ? "Moderate" : "High Risk";

  return (
    <div style={{ textAlign: "center" }}>
      <svg viewBox="0 0 180 105" style={{ width: "100%", maxWidth: "180px" }}>
        <path
          d={`M 20,88 A ${r},${r} 0 0 1 160,88`}
          fill="none"
          stroke="rgba(255,255,255,0.06)"
          strokeWidth="12"
          strokeLinecap="round"
        />
        {pct > 0 && (
          <path
            d={`M ${start.x},${start.y} A ${r},${r} 0 ${largeArc} 1 ${end.x},${end.y}`}
            fill="none"
            stroke={color}
            strokeWidth="12"
            strokeLinecap="round"
            style={{ filter: `drop-shadow(0 0 8px ${color})` }}
          />
        )}
        <text
          x="90"
          y="78"
          textAnchor="middle"
          fill={color}
          style={{ fontSize: "30px", fontWeight: 800, fontFamily: "Inter" }}
        >
          {pct.toFixed(1)}%
        </text>
      </svg>
      <div style={{ marginTop: "12px", color, fontWeight: 600, fontSize: "13px" }}>
        {label}
      </div>
    </div>
  );
}

// ─── ImportanceChart Component ────────────────────────────────────────────
function ImportanceChart({ data }) {
  if (!data) return null;
  const entries = Object.entries(data).sort((a, b) => b[1] - a[1]);
  const max = Math.max(...entries.map(e => e[1]), 1);

  return (
    <div>
      {entries.map(([label, value], i) => (
        <div key={label} style={{ marginBottom: "16px" }}>
          <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "6px" }}>
            <span style={{ fontSize: "12px", color: COLORS.slateLight, fontWeight: 500 }}>
              {label}
            </span>
          </div>
          <div
            style={{
              background: COLORS.navyMid,
              borderRadius: "4px",
              height: "32px",
              overflow: "hidden",
              position: "relative",
            }}
          >
            <div
              style={{
                height: "100%",
                width: `${(value / max) * 100}%`,
                background: BAR_COLORS[i % BAR_COLORS.length],
                display: "flex",
                alignItems: "center",
                justifyContent: value > 3 ? "center" : "flex-start",
                minWidth: value > 0 ? "32px" : "0",
                color: COLORS.white,
                fontSize: "11px",
                fontWeight: 600,
                transition: "width 0.3s ease",
              }}
            >
              {value > 3 ? `${value}%` : ""}
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

// ─── Main BoxOfficeOracle Component ───────────────────────────────────────
export default function BoxOfficeOracle() {
  const [director, setDirector] = useState("");
  const [producer, setProducer] = useState("");
  const [writer, setWriter] = useState("");
  const [genre, setGenre] = useState("Action");
  const [budget, setBudget] = useState(50_000_000);
  const [month, setMonth] = useState(6);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);

  function handlePredict(e) {
    e.preventDefault();
    if (!director) return;
    setLoading(true);
    setTimeout(() => {
      const result = predictMovieSuccess({
        director,
        producer: producer || "Unknown",
        writer: writer || "Unknown",
        genre,
        budget,
        month,
      });
      setPrediction(result);
      setLoading(false);
    }, 600);
  }

  const genres = Object.keys(GENRE_MULTIPLIERS);

  return (
    <div
      style={{
        background: COLORS.navy,
        color: COLORS.white,
        minHeight: "100vh",
        fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, sans-serif",
        WebkitFontSmoothing: "antialiased",
      }}
    >
      <div style={{ maxWidth: "1400px", margin: "0 auto", padding: "24px 32px" }}>
        {/* Header */}
        <header style={{ textAlign: "center", padding: "40px 0 32px" }}>
          <div style={{ display: "inline-flex", alignItems: "center", gap: "12px", marginBottom: "12px" }}>
            <div
              style={{
                width: "48px",
                height: "48px",
                background: `linear-gradient(135deg, ${COLORS.teal}, ${COLORS.violet})`,
                borderRadius: "14px",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                fontSize: "24px",
                boxShadow: `0 0 30px rgba(6, 182, 212, 0.3)`,
              }}
            >
              🎬
            </div>
            <h1
              style={{
                fontSize: "32px",
                fontWeight: 800,
                background: `linear-gradient(135deg, ${COLORS.teal}, #a78bfa)`,
                WebkitBackgroundClip: "text",
                WebkitTextFillColor: "transparent",
                margin: 0,
              }}
            >
              BoxOffice Oracle
            </h1>
          </div>
          <p style={{ color: COLORS.slateLight, fontSize: "15px", margin: "6px 0 0 0" }}>
            AI-powered movie financial success prediction
          </p>
        </header>

        {/* Main Grid */}
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "420px 1fr",
            gap: "28px",
            alignItems: "start",
          }}
        >
          {/* Left: Input Form */}
          <div>
            <div
              style={{
                background: COLORS.navyLight,
                border: `1px solid ${COLORS.slate}`,
                borderRadius: "12px",
                padding: "28px",
                boxShadow: "0 4px 24px rgba(0,0,0,0.3)",
              }}
            >
              <div
                style={{
                  fontSize: "16px",
                  fontWeight: 700,
                  color: COLORS.white,
                  marginBottom: "20px",
                  display: "flex",
                  alignItems: "center",
                  gap: "8px",
                }}
              >
                <div
                  style={{
                    width: "28px",
                    height: "28px",
                    background: COLORS.tealGlow,
                    borderRadius: "8px",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    fontSize: "14px",
                  }}
                >
                  🎯
                </div>
                Prediction Input
              </div>

              <form onSubmit={handlePredict}>
                <SearchableSelect
                  label="Director"
                  required
                  options={DIRECTORS}
                  placeholder="Search directors..."
                  value={director}
                  onChange={setDirector}
                />
                <SearchableSelect
                  label="Producer"
                  options={PRODUCERS}
                  placeholder="Search producers..."
                  value={producer}
                  onChange={setProducer}
                />
                <SearchableSelect
                  label="Writer"
                  options={WRITERS}
                  placeholder="Search writers..."
                  value={writer}
                  onChange={setWriter}
                />

                <div style={{ marginBottom: "18px" }}>
                  <label
                    style={{
                      display: "block",
                      fontSize: "12px",
                      fontWeight: 600,
                      color: COLORS.slateLight,
                      textTransform: "uppercase",
                      letterSpacing: "0.5px",
                      marginBottom: "6px",
                    }}
                  >
                    Genre
                  </label>
                  <select
                    value={genre}
                    onChange={e => setGenre(e.target.value)}
                    style={{
                      width: "100%",
                      background: COLORS.navy,
                      border: `1px solid ${COLORS.slate}`,
                      borderRadius: "8px",
                      padding: "10px 14px",
                      color: COLORS.white,
                      fontSize: "14px",
                      fontFamily: "inherit",
                      outline: "none",
                      appearance: "none",
                      backgroundImage: `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' fill='${COLORS.slateLight.slice(1)}' viewBox='0 0 16 16'%3E%3Cpath d='M8 11L3 6h10z'/%3E%3C/svg%3E")`,
                      backgroundRepeat: "no-repeat",
                      backgroundPosition: "right 14px center",
                      paddingRight: "40px",
                      cursor: "pointer",
                    }}
                  >
                    {genres.map(g => (
                      <option key={g} value={g}>
                        {g}
                      </option>
                    ))}
                  </select>
                </div>

                <div style={{ marginBottom: "18px" }}>
                  <label
                    style={{
                      display: "block",
                      fontSize: "12px",
                      fontWeight: 600,
                      color: COLORS.slateLight,
                      textTransform: "uppercase",
                      letterSpacing: "0.5px",
                      marginBottom: "6px",
                    }}
                  >
                    Production Budget
                  </label>
                  <div
                    style={{
                      textAlign: "center",
                      fontSize: "22px",
                      fontWeight: 700,
                      color: COLORS.teal,
                      marginBottom: "8px",
                    }}
                  >
                    {formatMoney(budget)}
                  </div>
                  <input
                    type="range"
                    min={500_000}
                    max={400_000_000}
                    step={500_000}
                    value={budget}
                    onChange={e => setBudget(Number(e.target.value))}
                    style={{
                      width: "100%",
                      height: "6px",
                      background: `linear-gradient(90deg, ${COLORS.emerald}, ${COLORS.teal}, ${COLORS.violet}, ${COLORS.rose})`,
                      borderRadius: "3px",
                      outline: "none",
                      WebkitAppearance: "none",
                      appearance: "none",
                    }}
                  />
                  <div
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      marginTop: "8px",
                      fontSize: "10px",
                      color: COLORS.slateLight,
                    }}
                  >
                    <span>$500K</span>
                    <span>$50M</span>
                    <span>$150M</span>
                    <span>$400M</span>
                  </div>
                  <div style={{ textAlign: "center", marginTop: "4px" }}>
                    <span style={{ fontSize: "11px", color: COLORS.teal, fontWeight: 600 }}>
                      {getBudgetTier(budget)} Budget
                    </span>
                  </div>
                </div>

                <button
                  type="button"
                  onClick={() => setShowAdvanced(!showAdvanced)}
                  style={{
                    background: "transparent",
                    border: "none",
                    color: COLORS.teal,
                    fontSize: "13px",
                    fontWeight: 600,
                    cursor: "pointer",
                    display: "flex",
                    alignItems: "center",
                    gap: "8px",
                    marginBottom: "18px",
                    padding: 0,
                  }}
                >
                  <span style={{ transform: showAdvanced ? "rotate(90deg)" : "rotate(0deg)", transition: "transform 0.3s" }}>
                    ▶
                  </span>
                  Advanced Options
                </button>

                {showAdvanced && (
                  <div style={{ marginBottom: "18px" }}>
                    <label
                      style={{
                        display: "block",
                        fontSize: "12px",
                        fontWeight: 600,
                        color: COLORS.slateLight,
                        textTransform: "uppercase",
                        letterSpacing: "0.5px",
                        marginBottom: "6px",
                      }}
                    >
                      Release Month
                    </label>
                    <select
                      value={month}
                      onChange={e => setMonth(Number(e.target.value))}
                      style={{
                        width: "100%",
                        background: COLORS.navy,
                        border: `1px solid ${COLORS.slate}`,
                        borderRadius: "8px",
                        padding: "10px 14px",
                        color: COLORS.white,
                        fontSize: "14px",
                        fontFamily: "inherit",
                        outline: "none",
                        appearance: "none",
                        backgroundImage: `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' fill='${COLORS.slateLight.slice(1)}' viewBox='0 0 16 16'%3E%3Cpath d='M8 11L3 6h10z'/%3E%3C/svg%3E")`,
                        backgroundRepeat: "no-repeat",
                        backgroundPosition: "right 14px center",
                        paddingRight: "40px",
                        cursor: "pointer",
                      }}
                    >
                      {MONTH_NAMES.slice(1).map((m, i) => (
                        <option key={i + 1} value={i + 1}>
                          {m}
                        </option>
                      ))}
                    </select>
                  </div>
                )}

                <button
                  type="submit"
                  disabled={!director || loading}
                  style={{
                    width: "100%",
                    background: !director || loading ? COLORS.slate : `linear-gradient(135deg, ${COLORS.teal}, ${COLORS.violet})`,
                    border: "none",
                    borderRadius: "8px",
                    padding: "12px 20px",
                    color: COLORS.white,
                    fontSize: "14px",
                    fontWeight: 600,
                    cursor: !director || loading ? "not-allowed" : "pointer",
                    transition: "all 0.3s ease",
                    boxShadow: !director || loading ? "none" : `0 0 20px ${COLORS.tealGlow}`,
                  }}
                >
                  {loading ? (
                    <span>
                      Analyzing
                      <span style={{ animation: "pulse 1.5s infinite" }}> .</span>
                      <span style={{ animation: "pulse 1.5s infinite 0.3s" }}> .</span>
                      <span style={{ animation: "pulse 1.5s infinite 0.6s" }}> .</span>
                    </span>
                  ) : (
                    "🔮  Predict Financial Outcome"
                  )}
                </button>
              </form>
            </div>
          </div>

          {/* Right: Results Dashboard */}
          <div>
            {!prediction && !loading ? (
              <div
                style={{
                  background: COLORS.navyLight,
                  border: `1px solid ${COLORS.slate}`,
                  borderRadius: "12px",
                  padding: "40px 28px",
                  textAlign: "center",
                  boxShadow: "0 4px 24px rgba(0,0,0,0.3)",
                }}
              >
                <div style={{ fontSize: "48px", marginBottom: "16px" }}>🎬</div>
                <div style={{ fontSize: "18px", fontWeight: 700, marginBottom: "8px" }}>
                  Ready to Predict
                </div>
                <div style={{ color: COLORS.slateLight, fontSize: "14px" }}>
                  Select a director and configure your film parameters, then click
                  "Predict Financial Outcome" to see AI-powered projections.
                </div>
              </div>
            ) : prediction ? (
              <div>
                <div
                  style={{
                    display: "flex",
                    gap: "8px",
                    flexWrap: "wrap",
                    marginBottom: "20px",
                  }}
                >
                  {[
                    { emoji: "🎬", text: prediction.input.director },
                    prediction.input.producer !== "Unknown" && { emoji: "🎥", text: prediction.input.producer },
                    prediction.input.writer !== "Unknown" && { emoji: "✍️", text: prediction.input.writer },
                    { emoji: "🎭", text: prediction.input.genre },
                    { emoji: "💰", text: formatMoney(prediction.input.budget) },
                    { emoji: "📅", text: MONTH_NAMES[prediction.input.release_month] },
                  ]
                    .filter(Boolean)
                    .map((chip, i) => (
                      <span
                        key={i}
                        style={{
                          background: COLORS.navyMid,
                          border: `1px solid ${COLORS.slate}`,
                          borderRadius: "20px",
                          padding: "6px 12px",
                          fontSize: "12px",
                          fontWeight: 500,
                          whiteSpace: "nowrap",
                        }}
                      >
                        {chip.emoji} {chip.text}
                      </span>
                    ))}
                </div>

                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))",
                    gap: "20px",
                  }}
                >
                  {/* Revenue Card */}
                  <div
                    style={{
                      background: COLORS.navyLight,
                      border: `1px solid ${COLORS.slate}`,
                      borderRadius: "12px",
                      padding: "28px",
                      boxShadow: "0 4px 24px rgba(0,0,0,0.3)",
                      animation: "slideIn 0.5s ease 0.1s both",
                    }}
                  >
                    <div
                      style={{
                        fontSize: "16px",
                        fontWeight: 700,
                        color: COLORS.white,
                        marginBottom: "20px",
                        display: "flex",
                        alignItems: "center",
                        gap: "8px",
                      }}
                    >
                      <div
                        style={{
                          width: "28px",
                          height: "28px",
                          background: COLORS.tealGlow,
                          borderRadius: "8px",
                          display: "flex",
                          alignItems: "center",
                          justifyContent: "center",
                          fontSize: "14px",
                        }}
                      >
                        💰
                      </div>
                      Predicted Revenue
                    </div>
                    <div style={{ fontSize: "28px", fontWeight: 700, color: COLORS.teal, marginBottom: "8px" }}>
                      {formatMoney(prediction.predicted_revenue)}
                    </div>
                    <div style={{ color: COLORS.slateLight, fontSize: "13px", marginBottom: "12px" }}>
                      Worldwide Gross
                    </div>
                    <div style={{ fontSize: "12px", color: COLORS.slateLight }}>
                      Range: <strong>{formatMoney(prediction.revenue_low)}</strong> —{" "}
                      <strong>{formatMoney(prediction.revenue_high)}</strong>
                    </div>
                  </div>

                  {/* ROI Card */}
                  <div
                    style={{
                      background: COLORS.navyLight,
                      border: `1px solid ${COLORS.slate}`,
                      borderRadius: "12px",
                      padding: "28px",
                      boxShadow: "0 4px 24px rgba(0,0,0,0.3)",
                      animation: "slideIn 0.5s ease 0.2s both",
                    }}
                  >
                    <div
                      style={{
                        fontSize: "16px",
                        fontWeight: 700,
                        color: COLORS.white,
                        marginBottom: "20px",
                        display: "flex",
                        alignItems: "center",
                        gap: "8px",
                      }}
                    >
                      <div
                        style={{
                          width: "28px",
                          height: "28px",
                          background: COLORS.tealGlow,
                          borderRadius: "8px",
                          display: "flex",
                          alignItems: "center",
                          justifyContent: "center",
                          fontSize: "14px",
                        }}
                      >
                        📈
                      </div>
                      Return on Investment
                    </div>
                    <div
                      style={{
                        fontSize: "28px",
                        fontWeight: 700,
                        color: getRoiColor(prediction.roi_percent),
                        marginBottom: "8px",
                      }}
                    >
                      {prediction.roi_percent > 0 ? "+" : ""}{prediction.roi_percent}%
                    </div>
                    <div style={{ color: COLORS.slateLight, fontSize: "13px", marginBottom: "12px" }}>
                      {prediction.roi_percent > 0 ? "Projected Profit" : "Projected Loss"}:{" "}
                      <strong style={{ color: getRoiColor(prediction.roi_percent) }}>
                        {formatMoney(Math.abs(prediction.predicted_revenue - prediction.budget))}
                      </strong>
                    </div>
                    <div style={{ fontSize: "12px", color: COLORS.slateLight }}>
                      Budget: <strong>{formatMoney(prediction.budget)}</strong>
                    </div>
                  </div>

                  {/* Success Gauge Card */}
                  <div
                    style={{
                      background: COLORS.navyLight,
                      border: `1px solid ${COLORS.slate}`,
                      borderRadius: "12px",
                      padding: "28px",
                      boxShadow: "0 4px 24px rgba(0,0,0,0.3)",
                      animation: "slideIn 0.5s ease 0.3s both",
                    }}
                  >
                    <div
                      style={{
                        fontSize: "16px",
                        fontWeight: 700,
                        color: COLORS.white,
                        marginBottom: "20px",
                        display: "flex",
                        alignItems: "center",
                        gap: "8px",
                      }}
                    >
                      <div
                        style={{
                          width: "28px",
                          height: "28px",
                          background: COLORS.tealGlow,
                          borderRadius: "8px",
                          display: "flex",
                          alignItems: "center",
                          justifyContent: "center",
                          fontSize: "14px",
                        }}
                      >
                        🎯
                      </div>
                      Success Probability
                    </div>
                    <div style={{ display: "flex", justifyContent: "center" }}>
                      <Gauge value={prediction.success_probability} />
                    </div>
                    <div style={{ textAlign: "center", marginTop: "12px" }}>
                      <span
                        style={{
                          display: "inline-block",
                          background: COLORS.navyMid,
                          border: `1px solid ${COLORS.slate}`,
                          borderRadius: "20px",
                          padding: "6px 12px",
                          fontSize: "12px",
                          fontWeight: 600,
                          color: COLORS.teal,
                        }}
                      >
                        {prediction.confidence === "High"
                          ? "●"
                          : prediction.confidence === "Medium"
                          ? "◐"
                          : "○"}{" "}
                        {prediction.confidence} Confidence
                      </span>
                    </div>
                  </div>

                  {/* Feature Importance Card */}
                  <div
                    style={{
                      background: COLORS.navyLight,
                      border: `1px solid ${COLORS.slate}`,
                      borderRadius: "12px",
                      padding: "28px",
                      boxShadow: "0 4px 24px rgba(0,0,0,0.3)",
                      animation: "slideIn 0.5s ease 0.4s both",
                      gridColumn: "1 / -1",
                    }}
                  >
                    <div
                      style={{
                        fontSize: "16px",
                        fontWeight: 700,
                        color: COLORS.white,
                        marginBottom: "20px",
                        display: "flex",
                        alignItems: "center",
                        gap: "8px",
                      }}
                    >
                      <div
                        style={{
                          width: "28px",
                          height: "28px",
                          background: COLORS.tealGlow,
                          borderRadius: "8px",
                          display: "flex",
                          alignItems: "center",
                          justifyContent: "center",
                          fontSize: "14px",
                        }}
                      >
                        🧠
                      </div>
                      Prediction Drivers
                    </div>
                    <ImportanceChart data={prediction.feature_importance} />
                  </div>
                </div>
              </div>
            ) : null}
          </div>
        </div>

        {/* Footer */}
        <footer style={{ textAlign: "center", padding: "40px 0 24px", color: COLORS.slate, fontSize: "12px" }}>
          BoxOffice Oracle — Predictions are statistical estimates based on historical patterns.
          Not financial advice.
        </footer>
      </div>

      <style>{`
        @keyframes slideIn {
          from {
            opacity: 0;
            transform: translateY(10px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
        
        @keyframes pulse {
          0%, 100% { opacity: 0.3; }
          50% { opacity: 1; }
        }

        ::-webkit-scrollbar {
          width: 6px;
        }
        ::-webkit-scrollbar-track {
          background: ${COLORS.navy};
        }
        ::-webkit-scrollbar-thumb {
          background: ${COLORS.navyMid};
          border-radius: 3px;
        }
      `}</style>
    </div>
  );
}
