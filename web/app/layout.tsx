import type { ReactElement, ReactNode } from "react";

/** Required document shell for the future demo UI. */
export default function RootLayout({
  children,
}: Readonly<{ children: ReactNode }>): ReactElement {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
