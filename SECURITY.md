# Security and data handling

`download` contacts the fixed NASA-linked URL. `run` reads local files offline. There is no telemetry, cloud model, login, or equipment connection.

Inputs/downloads have size bounds. Named archive members are read only after checksum verification; archive paths are never extracted. Reports use a new folder. Different existing data and output folders are not overwritten. Custom data are marked unverified.

Do not upload private equipment data or secrets in an issue. Use GitHub private vulnerability reporting if enabled, or contact the maintainer through the public profile without posting sensitive inputs. Include the version, minimal synthetic reproducer, and impact.

This research prototype is not for flight, shutdown, or operational safety decisions. Its maintenance labels are illustrative.
