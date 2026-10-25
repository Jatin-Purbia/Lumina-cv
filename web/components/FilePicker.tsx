"use client";

import type { ChangeEvent, ReactElement } from "react";
import { useId } from "react";

export interface FilePickerProps {
  accept: readonly string[];
  label: string;
  onPick: (file: File) => void;
  disabled?: boolean;
  variant?: "primary" | "quiet";
}

/** A real file input, visually hidden, with a label styled as a button. */
export function FilePicker({ accept, label, onPick, disabled = false, variant = "quiet" }: FilePickerProps): ReactElement {
  const id = useId();
  const onChange = (event: ChangeEvent<HTMLInputElement>): void => {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (file !== undefined) onPick(file);
  };
  return (
    <span className="picker">
      <input
        id={id}
        className="picker__input"
        type="file"
        accept={accept.join(",")}
        onChange={onChange}
        disabled={disabled}
      />
      <label htmlFor={id} className={`btn btn--${variant}${disabled ? " is-disabled" : ""}`}>
        {label}
      </label>
    </span>
  );
}
