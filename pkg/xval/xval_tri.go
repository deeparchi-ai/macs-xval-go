// Package xval — MACS §5 Tri-Model Cross-Validation
//
// Tri-model architecture follows banking "两地三中心" model applied to LLM vendors:
//   Primary Model (Production) → Audit Model (Hot Standby) → Tertiary Model (Cold Standby)
//
// Adjudication:
//   3/3 Consensus  → L1 Auto-Accept
//   2/3 Majority   → L2 Flagged + Execute
//   1/3 or 0/3    → L3 Escalate to Human
package xval

import (
	"fmt"
)

// ── Tri-model types ──

// TriVerdict captures the 4-way tri-model adjudication outcome.
type TriVerdict int

const (
	TriConsensus TriVerdict = iota // 3/3 — all three models agree
	TriMajority                    // 2/3 — two agree, one dissents
	TriMinority                    // 1/3 — only one agrees (or one viable)
	TriStalemate                   // 0/3 — all disagree or all failed
)

func (tv TriVerdict) String() string {
	switch tv {
	case TriConsensus:
		return "tri-consensus"
	case TriMajority:
		return "tri-majority"
	case TriMinority:
		return "tri-minority"
	case TriStalemate:
		return "tri-stalemate"
	default:
		return fmt.Sprintf("tri-unknown(%d)", tv)
	}
}

// TriModelPair defines a three-model pairing with vendor diversity constraint.
type TriModelPair struct {
	PrimaryVendor  ModelVendor
	PrimaryModel   string
	AuditVendor    ModelVendor
	AuditModel     string
	TertiaryVendor ModelVendor
	TertiaryModel  string
}

// Validate checks the three-vendor constraint — all three must be different.
func (tp *TriModelPair) Validate() error {
	if tp.PrimaryVendor == "" || tp.AuditVendor == "" || tp.TertiaryVendor == "" {
		return fmt.Errorf("all three vendors must be set")
	}
	if tp.PrimaryModel == "" || tp.AuditModel == "" || tp.TertiaryModel == "" {
		return fmt.Errorf("all three models must be set")
	}
	// Check all pairwise distinct
	vendors := map[ModelVendor]bool{tp.PrimaryVendor: true, tp.AuditVendor: true, tp.TertiaryVendor: true}
	if len(vendors) < 3 {
		return fmt.Errorf("all three vendors must be different: primary=%s audit=%s tertiary=%s",
			tp.PrimaryVendor, tp.AuditVendor, tp.TertiaryVendor)
	}
	return nil
}

// TriAdjudicationResult captures the outcome of tri-model cross-validation.
type TriAdjudicationResult struct {
	Verdict           TriVerdict
	PrimaryConfidence float64
	AuditConfidence   float64
	TertiaryConfidence float64

	// Agreement tracking
	PrimaryAgreesAudit    bool
	PrimaryAgreesTertiary bool
	AuditAgreesTertiary   bool

	// Majority tracking: which models form the majority
	MajorityModels  []string // model names in the agreeing group
	MinorityModels  []string // dissenting model names

	DisagreedSections []string
	EscalationReason  string

	// Whether the adjudication ran in degraded mode
	Degraded bool
}

// ── Degraded Mode / Failover ──

// FailoverState tracks which models are currently available for adjudication.
type FailoverState struct {
	PrimaryAvailable  bool
	AuditAvailable    bool
	TertiaryAvailable bool
	Mode              FailoverMode
}

// FailoverMode describes the current XVal degradation level.
type FailoverMode int

const (
	FailoverFull     FailoverMode = iota // all three available
	FailoverTwoModel                     // 2 models (one promoted)
	FailoverSingleModel                  // only 1 model (L0 self-critique)
	FailoverNone                         // all down → Warden triggers global pause
)

func (fm FailoverMode) String() string {
	switch fm {
	case FailoverFull:
		return "full-tri-model"
	case FailoverTwoModel:
		return "two-model-degraded"
	case FailoverSingleModel:
		return "single-model-L0"
	case FailoverNone:
		return "all-down"
	default:
		return fmt.Sprintf("unknown-failover(%d)", fm)
	}
}

// AvailableModelCount returns how many models are currently available.
func (fs *FailoverState) AvailableModelCount() int {
	n := 0
	if fs.PrimaryAvailable {
		n++
	}
	if fs.AuditAvailable {
		n++
	}
	if fs.TertiaryAvailable {
		n++
	}
	return n
}

// DetermineFailover computes the failover mode from available models.
func DetermineFailover(fs *FailoverState) FailoverMode {
	count := fs.AvailableModelCount()
	switch count {
	case 3:
		return FailoverFull
	case 2:
		return FailoverTwoModel
	case 1:
		return FailoverSingleModel
	default:
		return FailoverNone
	}
}

// PromoteOnFailure handles vendor outage promotions.
// Returns the new failover state after promoting standby models.
func PromoteOnFailure(fs *FailoverState, failedVendor ModelVendor) *FailoverState {
	newFS := &FailoverState{
		PrimaryAvailable:  fs.PrimaryAvailable,
		AuditAvailable:    fs.AuditAvailable,
		TertiaryAvailable: fs.TertiaryAvailable,
	}

	if failedVendor == VendorAnthropic || newFS.PrimaryAvailable == false {
		// Match by checking which model is unavailable
	}
	// Apply promotion chain
	if !newFS.PrimaryAvailable && newFS.AuditAvailable {
		newFS.PrimaryAvailable = true
		newFS.AuditAvailable = newFS.TertiaryAvailable
		newFS.TertiaryAvailable = false
	} else if !newFS.AuditAvailable && newFS.TertiaryAvailable {
		newFS.AuditAvailable = true
		newFS.TertiaryAvailable = false
	}
	newFS.Mode = DetermineFailover(newFS)
	return newFS
}

// MarkUnavailable marks a vendor as unavailable and updates failover mode.
func (fs *FailoverState) MarkUnavailable(vendor ModelVendor) {
	switch vendor {
	case VendorAnthropic:
		fs.PrimaryAvailable = false
	case VendorDeepSeek:
		fs.AuditAvailable = false
	case VendorGoogle:
		fs.TertiaryAvailable = false
	case VendorOpenAI:
		fs.AuditAvailable = false
	}
	fs.Mode = DetermineFailover(fs)
}

// ── Tri-model Adjudication ──

// agreementThreshold is the max confidence difference for two models to agree.
const agreementThreshold = 0.15

// modelsAgree returns true if two confidences are within the agreement threshold.
func modelsAgree(a, b float64) bool {
	diff := a - b
	if diff < 0 {
		diff = -diff
	}
	return diff <= agreementThreshold
}

// TriAdjudicate performs the full tri-model adjudication.
//
// The adjudication matrix (from governance spec §5.4):
//
//	Primary | Audit | Tertiary | Verdict    | Action
//	--------|-------|----------|------------|-------
//	✓       | ✓     | ✓        | Consensus  | L1 auto-accept
//	✓       | ✓     | ✗        | Majority   | L2 flagged+execute
//	✓       | ✗     | ✓        | Majority   | L2 flagged+execute
//	✗       | ✓     | ✓        | Majority   | L2 flagged+execute
//	✓       | ✗     | ✗        | Minority   | L3 escalate
//	✗       | ✗     | ✗        | Stalemate  | L3 escalate
func TriAdjudicate(primaryConf, auditConf, tertiaryConf float64, sections []string) TriAdjudicationResult {
	result := TriAdjudicationResult{
		PrimaryConfidence:  primaryConf,
		AuditConfidence:    auditConf,
		TertiaryConfidence: tertiaryConf,
		DisagreedSections:  sections,
	}

	// Determine pairwise agreement
	priAudAgree := modelsAgree(primaryConf, auditConf)
	priTerAgree := modelsAgree(primaryConf, tertiaryConf)
	audTerAgree := modelsAgree(auditConf, tertiaryConf)

	result.PrimaryAgreesAudit = priAudAgree
	result.PrimaryAgreesTertiary = priTerAgree
	result.AuditAgreesTertiary = audTerAgree

	agreeCount := 0
	if priAudAgree {
		agreeCount++
	}
	if priTerAgree {
		agreeCount++
	}
	if audTerAgree {
		agreeCount++
	}

	switch {
	case priAudAgree && priTerAgree && audTerAgree:
		// 3/3 consensus — all three pairs agree
		result.Verdict = TriConsensus
		result.MajorityModels = []string{"primary", "audit", "tertiary"}

	case priAudAgree:
		// Primary + Audit agree, Tertiary dissents
		result.Verdict = TriMajority
		result.MajorityModels = []string{"primary", "audit"}
		result.MinorityModels = []string{"tertiary"}
		result.EscalationReason = fmt.Sprintf(
			"2/3 majority (primary+audit): tertiary confidence %.2f diverges", tertiaryConf)

	case priTerAgree:
		// Primary + Tertiary agree, Audit dissents
		result.Verdict = TriMajority
		result.MajorityModels = []string{"primary", "tertiary"}
		result.MinorityModels = []string{"audit"}
		result.EscalationReason = fmt.Sprintf(
			"2/3 majority (primary+tertiary): audit confidence %.2f diverges", auditConf)

	case audTerAgree:
		// Audit + Tertiary agree, Primary dissents — still majority but Primary is dissenting
		result.Verdict = TriMajority
		result.MajorityModels = []string{"audit", "tertiary"}
		result.MinorityModels = []string{"primary"}
		result.EscalationReason = fmt.Sprintf(
			"2/3 majority (audit+tertiary): primary confidence %.2f diverges", primaryConf)

	case agreeCount == 0 && priAudAgree == false && priTerAgree == false && audTerAgree == false:
		// 0/3 stalemate — no pair agrees
		result.Verdict = TriStalemate
		result.EscalationReason = fmt.Sprintf(
			"0/3 stalemate: primary=%.2f audit=%.2f tertiary=%.2f — no model pair agrees",
			primaryConf, auditConf, tertiaryConf)

	default:
		// 1/3 minority — only one pair agrees (edge case)
		result.Verdict = TriMinority
		result.EscalationReason = fmt.Sprintf(
			"1/3 minority: primary=%.2f audit=%.2f tertiary=%.2f — only one pair agrees",
			primaryConf, auditConf, tertiaryConf)
	}

	return result
}
