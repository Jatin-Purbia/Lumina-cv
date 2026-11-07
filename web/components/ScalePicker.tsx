import type { ReactElement } from "react";

export const SCALE_PRESETS: readonly number[] = [0.25, 0.5, 1];

export interface ScalePickerProps {
  /** Server default reported by /options. */
  defaultValue: number;
  /** Selected preset, or null for the server default. */
  value: number | null;
  onChange: (value: number | null) => void;
  disabled?: boolean;
}

export function ScalePicker({ defaultValue, value, onChange, disabled = false }: ScalePickerProps): ReactElement {
  const options: { key: string; label: string; value: number | null }[] = [
    { key: "default", label: `Default (${defaultValue})`, value: null },
    ...SCALE_PRESETS.map((p) => ({ key: String(p), label: String(p), value: p })),
  ];
  return (
    <fieldset className="scale" disabled={disabled}>
      <legend className="field__label">Estimation scale</legend>
      <div className="scale__options">
        {options.map((o) => (
          <label key={o.key} className="scale__option">
            <input
              type="radio"
              name="curve-scale"
              checked={value === o.value}
              onChange={() => onChange(o.value)}
            />
            <span>{o.label}</span>
          </label>
        ))}
      </div>
      <p className="field__hint">
        Lower values estimate the correction at a smaller size. The output keeps the image&apos;s resolution.
      </p>
    </fieldset>
  );
}
