import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import { CompareSlider } from "@/components/CompareSlider";

function setup(initial?: number) {
  render(<CompareSlider beforeSrc="blob:before" afterSrc="blob:after" initial={initial} />);
  return screen.getByRole("slider", { name: "Comparison divider" });
}

describe("CompareSlider", () => {
  it("exposes slider semantics and both images", () => {
    const slider = setup();
    expect(slider).toHaveAttribute("aria-valuemin", "0");
    expect(slider).toHaveAttribute("aria-valuemax", "100");
    expect(slider).toHaveAttribute("aria-valuenow", "50");
    expect(slider).toHaveAttribute("aria-valuetext", "50% original, 50% enhanced");
    expect(slider).toHaveAttribute("tabindex", "0");
    expect(screen.getByAltText("Original photo")).toHaveAttribute("src", "blob:before");
    expect(screen.getByAltText("Enhanced photo")).toHaveAttribute("src", "blob:after");
  });

  it("moves with arrow keys, Shift for large steps", async () => {
    const user = userEvent.setup();
    const slider = setup();
    slider.focus();
    await user.keyboard("{ArrowRight}");
    expect(slider).toHaveAttribute("aria-valuenow", "52");
    await user.keyboard("{ArrowLeft}{ArrowLeft}");
    expect(slider).toHaveAttribute("aria-valuenow", "48");
    await user.keyboard("{Shift>}{ArrowRight}{/Shift}");
    expect(slider).toHaveAttribute("aria-valuenow", "58");
    await user.keyboard("{ArrowUp}");
    expect(slider).toHaveAttribute("aria-valuenow", "60");
    await user.keyboard("{ArrowDown}");
    expect(slider).toHaveAttribute("aria-valuenow", "58");
  });

  it("supports Home, End, PageUp and PageDown", async () => {
    const user = userEvent.setup();
    const slider = setup();
    slider.focus();
    await user.keyboard("{Home}");
    expect(slider).toHaveAttribute("aria-valuenow", "0");
    await user.keyboard("{PageUp}");
    expect(slider).toHaveAttribute("aria-valuenow", "10");
    await user.keyboard("{End}");
    expect(slider).toHaveAttribute("aria-valuenow", "100");
    await user.keyboard("{PageDown}");
    expect(slider).toHaveAttribute("aria-valuenow", "90");
  });

  it("clamps at both ends", async () => {
    const user = userEvent.setup();
    const slider = setup(1);
    slider.focus();
    await user.keyboard("{ArrowLeft}{ArrowLeft}");
    expect(slider).toHaveAttribute("aria-valuenow", "0");
    await user.keyboard("{End}{ArrowRight}");
    expect(slider).toHaveAttribute("aria-valuenow", "100");
  });

  it("does not swallow unrelated keys", () => {
    const slider = setup();
    const notPrevented = fireEvent.keyDown(slider, { key: "Tab" });
    expect(notPrevented).toBe(true);
    expect(slider).toHaveAttribute("aria-valuenow", "50");
    const prevented = fireEvent.keyDown(slider, { key: "ArrowRight" });
    expect(prevented).toBe(false);
  });
});
