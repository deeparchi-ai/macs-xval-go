package xval

import (
	"testing"
)

func TestVerdict_String(t *testing.T) {
	tests := []struct {
		v    Verdict
		want string
	}{
		{VerdictConsensus, "consensus"},
		{VerdictPartial, "partial"},
		{VerdictDisagree, "disagree"},
	}
	for _, tt := range tests {
		if got := tt.v.String(); got != tt.want {
			t.Errorf("Verdict(%d).String() = %q, want %q", tt.v, got, tt.want)
		}
	}
}

func TestModelPair_Validate(t *testing.T) {
	// Valid: different vendors
	pair := ModelPair{
		PrimaryVendor: VendorAnthropic,
		PrimaryModel:  "claude-sonnet-4",
		AuditVendor:   VendorDeepSeek,
		AuditModel:    "deepseek-v4-pro",
	}
	if err := pair.Validate(); err != nil {
		t.Errorf("expected valid pair, got: %v", err)
	}

	// Invalid: same vendor
	pair2 := ModelPair{
		PrimaryVendor: VendorAnthropic,
		PrimaryModel:  "claude-opus-4",
		AuditVendor:   VendorAnthropic,
		AuditModel:    "claude-sonnet-4",
	}
	if err := pair2.Validate(); err == nil {
		t.Error("expected error for same-vendor pair")
	}

	// Invalid: empty vendor
	pair3 := ModelPair{PrimaryModel: "gpt-4o", AuditModel: "deepseek-v4"}
	if err := pair3.Validate(); err == nil {
		t.Error("expected error for empty vendor")
	}
}

func TestXValConfig_Validate(t *testing.T) {
	// Valid subjective config
	cfg := XValConfig{
		AgentClass: AgentClassSubjective,
		Models: ModelPair{
			PrimaryVendor: VendorAnthropic,
			PrimaryModel:  "claude-sonnet-4",
			AuditVendor:   VendorDeepSeek,
			AuditModel:    "deepseek-v4-pro",
		},
		AutoAcceptL1:    true,
		AuditSampleRate: 0.1,
	}
	if err := cfg.Validate(); err != nil {
		t.Errorf("expected valid config, got: %v", err)
	}

	// Subjective without audit model → error
	cfg2 := XValConfig{
		AgentClass: AgentClassSubjective,
		Models:     ModelPair{PrimaryModel: "claude-sonnet-4"},
	}
	if err := cfg2.Validate(); err == nil {
		t.Error("expected error for subjective without audit model")
	}

	// Objective without audit model → OK
	cfg3 := XValConfig{
		AgentClass: AgentClassObjective,
		Models:     ModelPair{PrimaryModel: "claude-sonnet-4"},
	}
	if err := cfg3.Validate(); err != nil {
		t.Errorf("objective agents don't need audit model, got: %v", err)
	}

	// Bad sample rate
	cfg4 := XValConfig{
		AgentClass: AgentClassSubjective,
		Models: ModelPair{
			PrimaryVendor: VendorAnthropic, PrimaryModel: "c",
			AuditVendor: VendorDeepSeek, AuditModel: "d",
		},
		AuditSampleRate: 1.5,
	}
	if err := cfg4.Validate(); err == nil {
		t.Error("expected error for sample rate > 1.0")
	}
}

func TestAdjudicate_Consensus(t *testing.T) {
	result := Adjudicate(0.92, 0.88, nil)
	if result.Verdict != VerdictConsensus {
		t.Errorf("expected consensus, got %s", result.Verdict)
	}
}

func TestAdjudicate_ConsensusBoundary(t *testing.T) {
	// Just inside consensus boundary: diff=0.14, primary=0.80
	result := Adjudicate(0.80, 0.94, nil)
	if result.Verdict != VerdictConsensus {
		t.Errorf("expected consensus, got %s (primary=%.2f audit=%.2f diff=%.2f)",
			result.Verdict, result.PrimaryConfidence, result.AuditConfidence,
			result.PrimaryConfidence-result.AuditConfidence)
	}
}

func TestAdjudicate_Partial(t *testing.T) {
	sections := []string{"implementation approach", "technology choice"}
	result := Adjudicate(0.75, 0.50, sections)
	if result.Verdict != VerdictPartial {
		t.Errorf("expected partial, got %s", result.Verdict)
	}
	if len(result.DisagreedSections) != 2 {
		t.Errorf("expected 2 disagreed sections, got %d", len(result.DisagreedSections))
	}
}

func TestAdjudicate_Disagree(t *testing.T) {
	result := Adjudicate(0.90, 0.30, []string{"core architecture"})
	if result.Verdict != VerdictDisagree {
		t.Errorf("expected disagree, got %s", result.Verdict)
	}
	if result.EscalationReason == "" {
		t.Error("Disagree must have escalation reason")
	}
}

func TestEscalate(t *testing.T) {
	// L1: consensus + auto-accept
	cr := AdjudicationResult{Verdict: VerdictConsensus}
	if level := Escalate(cr, true); level != EscalationL1AutoAccept {
		t.Errorf("consensus+auto-accept → L1, got %s", level)
	}

	// L2: consensus without auto-accept
	if level := Escalate(cr, false); level != EscalationL2Flagged {
		t.Errorf("consensus without auto-accept → L2, got %s", level)
	}

	// L2: partial
	pr := AdjudicationResult{Verdict: VerdictPartial}
	if level := Escalate(pr, true); level != EscalationL2Flagged {
		t.Errorf("partial → L2, got %s", level)
	}

	// L3: disagree
	dr := AdjudicationResult{Verdict: VerdictDisagree, EscalationReason: "fundamental"}
	if level := Escalate(dr, true); level != EscalationL3Human {
		t.Errorf("disagree → L3, got %s", level)
	}
}

func TestDefaultConfig(t *testing.T) {
	cfg := DefaultConfig(AgentClassSubjective)
	if !cfg.AutoAcceptL1 {
		t.Error("default should auto-accept L1")
	}
	if cfg.AuditSampleRate != 0.1 {
		t.Errorf("default sample rate = %f, want 0.1", cfg.AuditSampleRate)
	}
}

func TestAgentClass_String(t *testing.T) {
	if AgentClassObjective.String() != "objective" {
		t.Errorf("Objective = %q", AgentClassObjective.String())
	}
	if AgentClassSubjective.String() != "subjective" {
		t.Errorf("Subjective = %q", AgentClassSubjective.String())
	}
}

func TestEscalationLevel_String(t *testing.T) {
	if EscalationL1AutoAccept.String() != "L1-auto-accept" {
		t.Errorf("L1 = %q", EscalationL1AutoAccept.String())
	}
	if EscalationL3Human.String() != "L3-human-escalation" {
		t.Errorf("L3 = %q", EscalationL3Human.String())
	}
}
