# macs-xval-go

MACS §5: Dual-model cross-validation for subjective agent outputs.

**Status:** v0.1 — 11 tests

## What

Subjective agents (architecture, strategy, product) produce opinions — not facts. A single model's output is not verifiable by computation. Cross-validation requires a second model from a DIFFERENT vendor.

Three-level adjudication:
- **Consensus**: both models agree → auto-accept (L1)
- **Partial**: agree on structure, disagree on details → flagged (L2)
- **Disagree**: fundamental disagreement → escalate to human (L3)

## Usage

```go
import "github.com/deeparchi-ai/macs-xval-go/pkg/xval"

cfg := xval.DefaultConfig(xval.AgentClassSubjective)
cfg.Models = xval.ModelPair{
    PrimaryVendor: xval.VendorAnthropic,
    PrimaryModel:  "claude-sonnet-4",
    AuditVendor:   xval.VendorDeepSeek,
    AuditModel:    "deepseek-v4-pro",
}

result := xval.Adjudicate(0.92, 0.88, nil)
level := xval.Escalate(result, cfg.AutoAcceptL1)
// → EscalationL1AutoAccept
```

## License

Apache 2.0 — zero dependencies (stdlib only).
