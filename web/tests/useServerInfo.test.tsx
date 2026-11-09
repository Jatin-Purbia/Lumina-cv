import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { getHealth, getOptions } from "@/lib/api";
import { useServerInfo } from "@/hooks/useServerInfo";
import type { OptionsResponse } from "@/lib/types";

vi.mock("@/lib/api", () => ({ getHealth: vi.fn(), getOptions: vi.fn() }));

beforeEach(() => {
  vi.resetAllMocks();
  vi.mocked(getHealth).mockResolvedValue({ status: "ok", model_loaded: true, device: "cpu", version: "test" });
  vi.mocked(getOptions).mockResolvedValue({ curve_scale: 1, max_pixels: 1000 });
});

describe("useServerInfo", () => {
  it("refreshes limits and defaults even when a restart happens between healthy polls", async () => {
    const { result } = renderHook(() => useServerInfo());
    await waitFor(() => expect(result.current.options?.max_pixels).toBe(1000));
    vi.mocked(getOptions).mockResolvedValue({ curve_scale: 0.5, max_pixels: 500 });
    act(() => result.current.refresh());
    await waitFor(() => expect(result.current.options).toEqual({ curve_scale: 0.5, max_pixels: 500 }));
    expect(result.current.health.phase).toBe("ready");
  });

  it("clears stale options on an outage and reloads them on recovery", async () => {
    const { result } = renderHook(() => useServerInfo());
    await waitFor(() => expect(result.current.options).not.toBeNull());
    vi.mocked(getHealth).mockRejectedValueOnce(new Error("offline"));
    act(() => result.current.refresh());
    await waitFor(() => expect(result.current.health.phase).toBe("unreachable"));
    expect(result.current.options).toBeNull();
    vi.mocked(getOptions).mockResolvedValue({ max_pixels: 2000 });
    act(() => result.current.refresh());
    await waitFor(() => expect(result.current.options?.max_pixels).toBe(2000));
  });

  it("ignores stale option requests after a newer refresh", async () => {
    let resolveOld: (value: OptionsResponse) => void = () => {};
    vi.mocked(getOptions).mockImplementationOnce(() => new Promise((resolve) => { resolveOld = resolve; }));
    const { result } = renderHook(() => useServerInfo());
    await waitFor(() => expect(getOptions).toHaveBeenCalledTimes(1));
    act(() => result.current.refresh());
    await waitFor(() => expect(result.current.options?.max_pixels).toBe(1000));
    await act(async () => { resolveOld({ max_pixels: 1 }); });
    expect(result.current.options?.max_pixels).toBe(1000);
  });
});
