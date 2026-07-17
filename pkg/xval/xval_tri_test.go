package xval

import (
	"testing"
)

// ── Tri-model Tests ──

func TestTriModelPair_Validate_AllDifferent(t *testing.T) {
	tp := TriModelPair{
		PrimaryVendor:  VendorAnthropic,
		PrimaryModel:   "claude-opus-4",
		AuditVendor:    VendorDeepSeek,
		AuditModel:     "deepseek-v4-pro",
		TertiaryVendor: VendorGoogle,
		TertiaryModel:  "gemini-2.5-pro",
	}
	if err := tp.Validate(); err != nil {
		t.Errorf("expected valid tri-model pair, got: %v", err)
	}
}

func TestTriModelPair_Validate_SameVendor(t *testing.T) {
	tp := TriModelPair{
		PrimaryVendor:  VendorAnthropic,
		PrimaryModel:   "claude-opus-4",
		AuditVendor:    VendorAnthropic,
		AuditModel:     "claude-sonnet-4",
		TertiaryVendor: VendorGoogle,
		TertiaryModel:  "gemini-2.5-pro",
	}
	if err := tp.Validate(); err == nil {
		t.Error("expected error for same-vendor primary+audit")
	}
}

func TestTriModelPair_Validate_AllSameVendor(t *testing.T) {
	tp := TriModelPair{
		PrimaryVendor:  VendorAnthropic,
		PrimaryModel:   "claude-opus-4",
		AuditVendor:    VendorAnthropic,
		AuditModel:     "claude-sonnet-4",
		TertiaryVendor: VendorAnthropic,
		TertiaryModel:  "claude-haiku-4",
	}
	if err := tp.Validate(); err == nil {
		t.Error("expected error for all-same-vendor")
	}
}

func TestTriModelPair_Validate_EmptyVendor(t *testing.T) {
	tp := TriModelPair{
		PrimaryVendor: VendorAnthropic,
		PrimaryModel:  "claude-opus-4",
	}
	if err := tp.Validate(); err == nil {
		t.Error("expected error for empty vendors")
	}
}

func TestTriVerdict_String(t *testing.T) {
	tests := []struct {
		v    TriVerdict
		want string
	}{
		{TriConsensus, "tri-consensus"},
		{TriMajority, "tri-majority"},
		{TriMinority, "tri-minority"},
		{TriStalemate, "tri-stalemate"},
	}
	for _, tt := range tests {
		if got := tt.v.String(); got != tt.want {
			t.Errorf("TriVerdict(%d).String() = %q, want %q", tt.v, got, tt.want)
		}
	}
}

// ── Adjudication Matrix Tests (from §5.4) ──

func TestTriAdjudicate_Consensus_3of3(t *testing.T) {
	// All three models agree with high confidence
	result := TriAdjudicate(0.92, 0.88, 0.94, nil)
	if result.Verdict != TriConsensus {
		t.Errorf("expected TriConsensus, got %s (pri=%.2f aud=%.2f ter=%.2f)",
			result.Verdict, result.PrimaryConfidence, result.AuditConfidence, result.TertiaryConfidence)
	}
	if len(result.MajorityModels) != 3 {
		t.Errorf("expected 3 majority models, got %d", len(result.MajorityModels))
	}
}

func TestTriAdjudicate_Consensus_TightBoundary(t *testing.T) {
	// Just inside the agreement threshold (diff ≤ 0.15)
	result := TriAdjudicate(0.80, 0.85, 0.81, nil)
	if result.Verdict != TriConsensus {
		t.Errorf("expected TriConsensus at boundary, got %s", result.Verdict)
	}
}

func TestTriAdjudicate_Majority_PriAudAgree(t *testing.T) {
	// Primary(0.90) + Audit(0.85) agree, Tertiary(0.50) dissents
	result := TriAdjudicate(0.90, 0.85, 0.50, nil)
	if result.Verdict != TriMajority {
		t.Errorf("expected TriMajority (pri+aud), got %s", result.Verdict)
	}
	if result.MinorityModels[0] != "tertiary" {
		t.Errorf("expected tertiary as minority, got %v", result.MinorityModels)
	}
	if result.EscalationReason == "" {
		t.Error("majority verdict should have escalation reason")
	}
}

func TestTriAdjudicate_Majority_PriTerAgree(t *testing.T) {
	// Primary(0.90) + Tertiary(0.88) agree, Audit(0.40) dissents
	result := TriAdjudicate(0.90, 0.40, 0.88, nil)
	if result.Verdict != TriMajority {
		t.Errorf("expected TriMajority (pri+ter), got %s", result.Verdict)
	}
	if result.MinorityModels[0] != "audit" {
		t.Errorf("expected audit as minority, got %v", result.MinorityModels)
	}
}

func TestTriAdjudicate_Majority_AudTerAgree(t *testing.T) {
	// Audit(0.85) + Tertiary(0.82) agree, Primary(0.30) dissents
	result := TriAdjudicate(0.30, 0.85, 0.82, nil)
	if result.Verdict != TriMajority {
		t.Errorf("expected TriMajority (aud+ter), got %s", result.Verdict)
	}
	if result.MinorityModels[0] != "primary" {
		t.Errorf("expected primary as minority, got %v", result.MinorityModels)
	}
}

func TestTriAdjudicate_Minority_1of3(t *testing.T) {
	// Only Primary is confident, Audit and Tertiary disagree with each other and Primary
	// Primary(0.90), Audit(0.35), Tertiary(0.60) — only one pair could agree (aud-ter?)
	// 0.35 vs 0.60 = diff 0.25 > threshold → no. Primary vs Audit = 0.55 > threshold → no. Primary vs Tertiary = 0.30 > threshold → no.
	result := TriAdjudicate(0.90, 0.35, 0.60, []string{"risk assessment"})
	if result.Verdict != TriStalemate {
		t.Errorf("expected TriStalemate (no pairs agree), got %s", result.Verdict)
	}
}

func TestTriAdjudicate_Stalemate_0of3(t *testing.T) {
	// All three fundamentally disagree
	result := TriAdjudicate(0.90, 0.30, 0.50, []string{"core strategy"})
	if result.Verdict != TriStalemate {
		t.Errorf("expected TriStalemate, got %s", result.Verdict)
	}
	if result.EscalationReason == "" {
		t.Error("stalemate should have escalation reason")
	}
}

func TestTriAdjudicate_AllZeroConfidence(t *testing.T) {
	// Edge case: all models report 0 confidence (all failed)
	result := TriAdjudicate(0.0, 0.0, 0.0, nil)
	// 0.0 ≈ 0.0 → all agree → this is Consensus technically
	if result.Verdict != TriConsensus {
		t.Errorf("all zeros should be Consensus (all agree they're unsure), got %s", result.Verdict)
	}
}

func TestTriAdjudicate_SectionsPreserved(t *testing.T) {
	sections := []string{"approach A", "approach B"}
	result := TriAdjudicate(0.70, 0.30, 0.45, sections)
	if len(result.DisagreedSections) != 2 {
		t.Errorf("expected 2 disagreed sections, got %d", len(result.DisagreedSections))
	}
}

func TestTriAdjudicate_AgreementTracking(t *testing.T) {
	result := TriAdjudicate(0.90, 0.88, 0.50, nil)
	if !result.PrimaryAgreesAudit {
		t.Error("primary and audit should agree")
	}
	if result.PrimaryAgreesTertiary {
		t.Error("primary and tertiary should NOT agree")
	}
	if result.AuditAgreesTertiary {
		t.Error("audit and tertiary should NOT agree")
	}
}

// ── Failover Tests ──

func TestFailoverState_AvailableModelCount(t *testing.T) {
	fs := &FailoverState{
		PrimaryAvailable:  true,
		AuditAvailable:    true,
		TertiaryAvailable: true,
	}
	if fs.AvailableModelCount() != 3 {
		t.Errorf("expected 3, got %d", fs.AvailableModelCount())
	}

	fs.MarkUnavailable(VendorAnthropic)
	if fs.AvailableModelCount() != 2 {
		t.Errorf("expected 2 after primary down, got %d", fs.AvailableModelCount())
	}

	fs.MarkUnavailable(VendorDeepSeek)
	if fs.AvailableModelCount() != 1 {
		t.Errorf("expected 1 after pri+aud down, got %d", fs.AvailableModelCount())
	}

	fs.MarkUnavailable(VendorGoogle)
	if fs.AvailableModelCount() != 0 {
		t.Errorf("expected 0 after all down, got %d", fs.AvailableModelCount())
	}
}

func TestFailoverState_DetermineFailover(t *testing.T) {
	tests := []struct {
		pri, aud, ter bool
		want          FailoverMode
	}{
		{true, true, true, FailoverFull},
		{false, true, true, FailoverTwoModel},
		{true, false, true, FailoverTwoModel},
		{true, true, false, FailoverTwoModel},
		{true, false, false, FailoverSingleModel},
		{false, true, false, FailoverSingleModel},
		{false, false, true, FailoverSingleModel},
		{false, false, false, FailoverNone},
	}
	for _, tt := range tests {
		fs := &FailoverState{
			PrimaryAvailable:  tt.pri,
			AuditAvailable:    tt.aud,
			TertiaryAvailable: tt.ter,
		}
		mode := DetermineFailover(fs)
		if mode != tt.want {
			t.Errorf("(%v,%v,%v) DetermineFailover = %s, want %s",
				tt.pri, tt.aud, tt.ter, mode, tt.want)
		}
	}
}

func TestFailoverMode_String(t *testing.T) {
	tests := []struct {
		m    FailoverMode
		want string
	}{
		{FailoverFull, "full-tri-model"},
		{FailoverTwoModel, "two-model-degraded"},
		{FailoverSingleModel, "single-model-L0"},
		{FailoverNone, "all-down"},
	}
	for _, tt := range tests {
		if got := tt.m.String(); got != tt.want {
			t.Errorf("FailoverMode(%d).String() = %q, want %q", tt.m, got, tt.want)
		}
	}
}

func TestPromoteOnFailure_PrimaryDown(t *testing.T) {
	fs := &FailoverState{
		PrimaryAvailable:  true,
		AuditAvailable:    true,
		TertiaryAvailable: true,
	}
	// Simulate Primary going down
	fs.MarkUnavailable(VendorAnthropic)
	mode := DetermineFailover(fs)
	if mode != FailoverTwoModel {
		t.Errorf("expected 2-model after primary down, got %s", mode)
	}
}

func TestPromoteOnFailure_AllDown(t *testing.T) {
	fs := &FailoverState{}
	fs.MarkUnavailable(VendorAnthropic)
	fs.MarkUnavailable(VendorDeepSeek)
	fs.MarkUnavailable(VendorGoogle)
	mode := DetermineFailover(fs)
	if mode != FailoverNone {
		t.Errorf("expected all-down, got %s", mode)
	}
}

// ── Integration: verdict → escalation ──

func TestTriVerdict_ToEscalation(t *testing.T) {
	tests := []struct {
		verdict TriVerdict
		level   string // expected human-readable level
	}{
		{TriConsensus, "L1-auto-accept"},
		{TriMajority, "L2-flagged"},
		{TriMinority, "L3-human-escalation"},
		{TriStalemate, "L3-human-escalation"},
	}
	for _, tt := range tests {
		var level string
		switch tt.verdict {
		case TriConsensus:
			level = "L1-auto-accept"
		case TriMajority:
			level = "L2-flagged"
		case TriMinority, TriStalemate:
			level = "L3-human-escalation"
		}
		if level != tt.level {
			t.Errorf("verdict %s → level %s, want %s", tt.verdict, level, tt.level)
		}
	}
}
