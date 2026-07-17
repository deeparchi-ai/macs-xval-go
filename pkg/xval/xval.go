// Package xval implements MACS §5: Dual-model cross-validation for
// subjective agent outputs.
//
// Subjective agents (architecture, strategy, product) produce opinions,
// not facts. A single model's output is not verifiable by computation.
// Cross-validation requires a second model from a DIFFERENT vendor to
// audit the primary model's output.
//
// Three-level adjudication:
//   - Consensus: both models agree → auto-accept
//   - Partial: agreement on structure, disagreement on details → flagged
//   - Disagree: fundamental disagreement → escalate to human
package xval

import (
	"fmt"
)

// Verdict is the result of cross-validation.
type Verdict int

const (
	VerdictConsensus Verdict = iota // both models agree
	VerdictPartial                  // agreement on structure, disagreement on details
	VerdictDisagree                 // fundamental disagreement
)

// String returns the verdict name.
func (v Verdict) String() string {
	switch v {
	case VerdictConsensus:
		return "consensus"
	case VerdictPartial:
		return "partial"
	case VerdictDisagree:
		return "disagree"
	default:
		return fmt.Sprintf("unknown(%d)", v)
	}
}

// AgentClass classifies whether an agent requires cross-validation.
type AgentClass int

const (
	AgentClassObjective  AgentClass = iota // output is mechanically verifiable (compiles, tests pass)
	AgentClassSubjective                   // output requires judgment (architecture, strategy)
)

// String returns the agent class name.
func (ac AgentClass) String() string {
	switch ac {
	case AgentClassObjective:
		return "objective"
	case AgentClassSubjective:
		return "subjective"
	default:
		return fmt.Sprintf("unknown(%d)", ac)
	}
}

// ModelVendor identifies a model vendor for the different-vendor constraint.
type ModelVendor string

const (
	VendorAnthropic ModelVendor = "anthropic"
	VendorOpenAI    ModelVendor = "openai"
	VendorDeepSeek  ModelVendor = "deepseek"
	VendorGoogle    ModelVendor = "google"
	VendorMeta      ModelVendor = "meta"
)

// ModelPair defines a primary and audit model pairing.
type ModelPair struct {
	PrimaryVendor ModelVendor
	PrimaryModel  string
	AuditVendor   ModelVendor
	AuditModel    string
}

// Validate checks the different-vendor constraint.
func (mp *ModelPair) Validate() error {
	if mp.PrimaryVendor == "" || mp.AuditVendor == "" {
		return fmt.Errorf("both primary and audit vendors must be set")
	}
	if mp.PrimaryVendor == mp.AuditVendor {
		return fmt.Errorf("primary and audit models must be from different vendors, got %s == %s",
			mp.PrimaryVendor, mp.AuditVendor)
	}
	return nil
}

// XValConfig configures the cross-validation engine.
type XValConfig struct {
	// AgentClass determines whether XVal is mandatory.
	AgentClass AgentClass

	// Models defines the model pair for cross-validation.
	Models ModelPair

	// AutoAcceptL1 enables automatic acceptance when both models reach consensus.
	// Default: true for production, false for development.
	AutoAcceptL1 bool

	// AuditSampleRate controls how often subjective agents without XVal
	// are spot-checked (0.0 = never, 1.0 = always). Used when XVal is
	// degraded (L0 mode).
	AuditSampleRate float64
}

// DefaultConfig returns a production-ready XVal configuration.
func DefaultConfig(agentClass AgentClass) XValConfig {
	return XValConfig{
		AgentClass:      agentClass,
		AutoAcceptL1:    true,
		AuditSampleRate: 0.1, // 10% spot-checks when XVal degraded
	}
}

// Validate checks the configuration for correctness.
func (c *XValConfig) Validate() error {
	if c.Models.PrimaryModel == "" {
		return fmt.Errorf("primary model must be set")
	}
	if c.AgentClass == AgentClassSubjective {
		if c.Models.AuditModel == "" {
			return fmt.Errorf("subjective agents require an audit model")
		}
		if err := c.Models.Validate(); err != nil {
			return err
		}
	}
	if c.AuditSampleRate < 0 || c.AuditSampleRate > 1 {
		return fmt.Errorf("audit sample rate must be between 0.0 and 1.0, got %f", c.AuditSampleRate)
	}
	return nil
}

// AdjudicationResult is the outcome of cross-validation.
type AdjudicationResult struct {
	Verdict              Verdict
	PrimaryConfidence    float64 // 0.0-1.0
	AuditConfidence      float64 // 0.0-1.0
	DisagreedSections    []string
	PrimaryReasoningHash string
	AuditReasoningHash   string
	EscalationReason     string // set when VerdictDisagree
}

// Adjudicate determines the verdict from two confidence scores.
func Adjudicate(primaryConfidence, auditConfidence float64, sections []string) AdjudicationResult {
	diff := primaryConfidence - auditConfidence
	if diff < 0 {
		diff = -diff
	}

	result := AdjudicationResult{
		PrimaryConfidence: primaryConfidence,
		AuditConfidence:   auditConfidence,
	}

	switch {
	case diff <= 0.15 && primaryConfidence >= 0.7:
		// Both models agree with high confidence
		result.Verdict = VerdictConsensus
	case diff <= 0.4:
		// Some disagreement but not fundamental
		result.Verdict = VerdictPartial
		result.DisagreedSections = sections
	default:
		// Fundamental disagreement
		result.Verdict = VerdictDisagree
		result.DisagreedSections = sections
		result.EscalationReason = fmt.Sprintf(
			"fundamental disagreement: primary=%.2f, audit=%.2f, diff=%.2f",
			primaryConfidence, auditConfidence, diff)
	}

	return result
}

// EscalationLevel maps the adjudication result to the three-tier escalation.
type EscalationLevel int

const (
	EscalationL1AutoAccept EscalationLevel = iota // consensus → auto-accept
	EscalationL2Flagged                          // partial → apply refinement
	EscalationL3Human                            // disagree → escalate
)

// String returns the escalation level name.
func (el EscalationLevel) String() string {
	switch el {
	case EscalationL1AutoAccept:
		return "L1-auto-accept"
	case EscalationL2Flagged:
		return "L2-flagged"
	case EscalationL3Human:
		return "L3-human-escalation"
	default:
		return fmt.Sprintf("unknown(%d)", el)
	}
}

// Escalate determines the escalation level from an adjudication result.
func Escalate(result AdjudicationResult, autoAcceptL1 bool) EscalationLevel {
	switch result.Verdict {
	case VerdictConsensus:
		if autoAcceptL1 {
			return EscalationL1AutoAccept
		}
		return EscalationL2Flagged
	case VerdictPartial:
		return EscalationL2Flagged
	case VerdictDisagree:
		return EscalationL3Human
	default:
		return EscalationL3Human
	}
}
